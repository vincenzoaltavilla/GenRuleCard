from __future__ import annotations

from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from ..core.artifacts import CorpusBundle, ProcessedDataset
from ..core.config import RuleCardConfig, VAEConfig
from ..rulecard_fit.encoding import make_real_row, padding_row, rows_to_leaves
from ..rulecard_fit.scorecards import import_rulecard, scorecard_from_leaves, scorecard_outputs
from .tree_matrix import decode_tree_output_to_rows


@dataclass
class DecodedScoreCard:
    rows: pd.DataFrame
    leaves: pd.DataFrame
    prior_log_odds: float
    info: dict
    scorecard: object


def _drop_logically_impossible_leaves(
    leaves: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Rimuove intervalli vuoti, indipendentemente dai dati osservati."""
    if leaves.empty or "conditions_key" not in leaves.columns:
        return leaves, 0

    def is_possible(path) -> bool:
        bounds: dict[int, dict[str, tuple[float, bool] | None]] = {}
        for feature, threshold, operator, *_ in path:
            feature = int(feature)
            threshold = float(threshold)
            operator = str(operator)
            bound = bounds.setdefault(feature, {"lower": None, "upper": None})
            if operator in (">", ">="):
                candidate = (threshold, operator == ">=")
                current = bound["lower"]
                if current is None or candidate[0] > current[0]:
                    bound["lower"] = candidate
                elif candidate[0] == current[0]:
                    bound["lower"] = (candidate[0], candidate[1] and current[1])
            elif operator in ("<", "<="):
                candidate = (threshold, operator == "<=")
                current = bound["upper"]
                if current is None or candidate[0] < current[0]:
                    bound["upper"] = candidate
                elif candidate[0] == current[0]:
                    bound["upper"] = (candidate[0], candidate[1] and current[1])
        for bound in bounds.values():
            lower, upper = bound["lower"], bound["upper"]
            if lower is None or upper is None:
                continue
            if lower[0] > upper[0]:
                return False
            if lower[0] == upper[0] and not (lower[1] and upper[1]):
                return False
        return True

    possible = leaves["conditions_key"].apply(is_possible).to_numpy(dtype=bool)
    removed = int((~possible).sum())
    return leaves.loc[possible].reset_index(drop=True), removed


class ConstrainedDecoder:
    def __init__(
        self,
        model,
        corpus: CorpusBundle,
        data: ProcessedDataset,
        config: RuleCardConfig,
        source_dir,
        vae_config: VAEConfig | None = None,
    ):
        self.model = model
        self.corpus = corpus
        self.data = data
        self.config = config
        self.vae_config = vae_config
        self.source_dir = source_dir
        self.device = next(model.parameters()).device
        self.next_scorecard_id = 50_000_000
        self.pattern_meta = corpus.pattern_vocab.set_index("pattern_id").sort_index()
        self.stump_ids_by_root = {
            int(root): group.index.to_numpy(dtype=np.int64)
            for root, group in self.pattern_meta.query("child_feat < 0").groupby("root_feat")
        }
        self.pair_ids_by_root = {
            int(root): group.index.to_numpy(dtype=np.int64)
            for root, group in self.pattern_meta.query("child_feat >= 0").groupby("root_feat")
        }
        threshold_grid: dict[int, list[float]] = {idx: [] for idx in range(len(data.feature_names))}
        for conditions in corpus.corpus_df["conditions_key"]:
            for feature, threshold, _ in conditions:
                threshold_grid[int(feature)].append(float(threshold))
        self.threshold_grid = {idx: np.unique(np.asarray(values, dtype=np.float32)) for idx, values in threshold_grid.items() if values}

    def _snap_threshold(self, normalized: float, feature: int) -> float:
        raw = self.data.feature_min[feature] + float(np.clip(normalized, 0.0, 1.0)) * self.data.feature_range[feature]
        grid = self.threshold_grid[feature]
        return float(grid[np.argmin(np.abs(grid - raw))])

    @staticmethod
    def _log_softmax(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        values = values - values.max(axis=-1, keepdims=True)
        return values - np.log(np.exp(values).sum(axis=-1, keepdims=True))

    def _build_row(self, scorecard_id: int, tree_idx: int, row_in_tree: int, kind: str, pattern_id: int, thr1: float, thr2: float, w: float, w_counter: float) -> dict:
        meta = self.pattern_meta.loc[int(pattern_id)]
        row = make_real_row(
            scorecard_id, tree_idx, row_in_tree, kind,
            int(meta["root_feat"]), int(meta["child_feat"]), thr1, thr2, w, w_counter,
            self.data.feature_names,
        )
        row["pattern_id"] = int(pattern_id)
        return row

    def _decode_rows(
        self,
        pattern_logits: np.ndarray,
        thresholds: np.ndarray,
        weights: np.ndarray,
        padding_logits: np.ndarray,
        scorecard_id: int,
        target_n_rules: int | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        log_pattern = self._log_softmax(pattern_logits)
        p_padding = np.clip(expit(padding_logits), 1e-8, 1.0 - 1e-8)
        log_padding, log_real = np.log(p_padding), np.log(1.0 - p_padding)
        raw_weights = weights * self.corpus.weight_std + self.corpus.weight_mean
        # Il corpus puo contenere soli stump, sole coppie oppure entrambi. Per
        # ogni posizione conserviamo separatamente il miglior albero reale e
        # il punteggio del padding: il prefisso attivo non deve mai scegliere
        # PAD per poi convertirlo artificialmente in uno stump.
        real_choices = []
        padding_scores = []
        for tree_idx in range(self.corpus.t_max):
            left, right = 2 * tree_idx, 2 * tree_idx + 1
            padding_scores.append(float(log_padding[left] + log_padding[right]))
            candidates = []
            for root, ids in self.stump_ids_by_root.items():
                pid = int(ids[np.argmax(log_pattern[left, ids])])
                candidates.append(("STUMP", log_real[left] + log_pattern[left, pid] + log_padding[right], (pid,)))
            for root, ids in self.pair_ids_by_root.items():
                left_pid = int(ids[np.argmax(log_pattern[left, ids])])
                right_pid = int(ids[np.argmax(log_pattern[right, ids])])
                candidates.append(("FULL", log_real[left] + log_pattern[left, left_pid] + log_real[right] + log_pattern[right, right_pid], (left_pid, right_pid)))
                if root in self.stump_ids_by_root:
                    stump = self.stump_ids_by_root[root]
                    left_leaf = int(stump[np.argmax(log_pattern[left, stump])])
                    right_leaf = int(stump[np.argmax(log_pattern[right, stump])])
                    candidates.append(("ASYM_L", log_real[left] + log_pattern[left, left_pid] + log_real[right] + log_pattern[right, right_leaf], (left_pid, right_leaf)))
                    candidates.append(("ASYM_R", log_real[left] + log_pattern[left, left_leaf] + log_real[right] + log_pattern[right, right_pid], (left_leaf, right_pid)))
            if not candidates:
                raise RuntimeError("Il vocabolario del corpus non contiene pattern decodificabili.")
            real_choices.append(max(candidates, key=lambda item: item[1]))
        prefix_scores = [0.0]
        prefix_rule_counts = [0]
        real_probability_by_tree = []
        for tree_idx in range(self.corpus.t_max):
            left, right = 2 * tree_idx, 2 * tree_idx + 1
            real_probability_by_tree.append(float(np.mean([1.0 - p_padding[left], 1.0 - p_padding[right]])))
        predicted_active = float(np.clip(np.sum(real_probability_by_tree), 1.0, self.corpus.t_max))
        count_penalty = float(getattr(self.vae_config, "decode_count_penalty", 0.0) if self.vae_config is not None else 0.0)
        running_rules = 0
        for n_active in range(1, self.corpus.t_max + 1):
            value = sum(real_choices[idx][1] for idx in range(n_active))
            value += sum(padding_scores[n_active:])
            running_rules += 1 if real_choices[n_active - 1][0] == "STUMP" else 2
            prefix_rule_counts.append(running_rules)
            if target_n_rules is None:
                value -= count_penalty * abs(float(n_active) - predicted_active)
            else:
                value -= count_penalty * abs(float(running_rules) - float(target_n_rules))
            prefix_scores.append(value)
        n_active = max(1, int(np.argmax(prefix_scores[1:]) + 1))
        rows = []
        for tree_idx in range(self.corpus.t_max):
            left, right = 2 * tree_idx, 2 * tree_idx + 1
            if tree_idx >= n_active:
                rows.extend([padding_row(scorecard_id, tree_idx, 0), padding_row(scorecard_id, tree_idx, 1)])
                continue
            kind, _, ids = real_choices[tree_idx]
            if kind == "STUMP":
                pid = ids[0]
                root = int(self.pattern_meta.loc[pid, "root_feat"])
                thr = self._snap_threshold(thresholds[left, 0], root)
                rows.append(self._build_row(scorecard_id, tree_idx, 0, "STUMP", pid, thr, 0.0, raw_weights[left, 0], raw_weights[left, 1]))
                rows.append(padding_row(scorecard_id, tree_idx, 1))
                continue
            left_pid, right_pid = ids
            root = int(self.pattern_meta.loc[left_pid, "root_feat"])
            root_thr = self._snap_threshold(np.mean([thresholds[left, 0], thresholds[right, 0]]), root)
            for pos, pid, label in [(0, left_pid, "L"), (1, right_pid, "R")]:
                index = left if pos == 0 else right
                meta = self.pattern_meta.loc[pid]
                if int(meta["child_feat"]) >= 0:
                    thr2 = self._snap_threshold(thresholds[index, 1], int(meta["child_feat"]))
                    w, wc, kind_name = raw_weights[index, 0], raw_weights[index, 1], label
                else:
                    thr2 = 0.0
                    w = wc = float(raw_weights[index].mean())
                    kind_name = f"{label}_LEAF"
                rows.append(self._build_row(scorecard_id, tree_idx, pos, kind_name, pid, root_thr, thr2, w, wc))
        return pd.DataFrame(rows), {
            "n_active": n_active,
            "n_rules_prefix": prefix_rule_counts[n_active],
            "target_n_rules": target_n_rules,
            "predicted_active": predicted_active,
            "decode_count_penalty": count_penalty,
            "prefix_scores": prefix_scores,
        }

    @torch.no_grad()
    def decode(
        self,
        z: np.ndarray,
        scorecard_id: int = -1,
        target_n_rules: int | None = None,
    ) -> DecodedScoreCard:
        z_tensor = torch.tensor(np.asarray(z, dtype=np.float32)[None], dtype=torch.float32, device=self.device)
        decoded = self.model.decode(z_tensor)
        if "type_logits" in decoded:
            candidate_id = int(scorecard_id)
            if candidate_id < 0:
                candidate_id = self.next_scorecard_id
                self.next_scorecard_id += 1
            output_np = {
                key: value[0].cpu().numpy()
                for key, value in decoded.items()
                if key != "prior"
            }
            rows, leaves, prior, info = decode_tree_output_to_rows(
                output_np,
                float(decoded["prior"][0].cpu().numpy()),
                candidate_id,
                self.corpus,
                self.data,
            )
            leaves, removed_impossible = _drop_logically_impossible_leaves(leaves)
            info["n_logically_impossible_removed"] = removed_impossible
            info["target_n_rules"] = target_n_rules
            _, _, scorecard_cls = import_rulecard(self.source_dir)
            scorecard = scorecard_from_leaves(leaves, prior, self.data.model_classes, self.config, scorecard_cls)
            info["n_rules_prefix"] = int(len(scorecard.scorecard))
            return DecodedScoreCard(rows, leaves, prior, info, scorecard)
        rows, info = self._decode_rows(
            decoded["pattern_logits"][0].cpu().numpy(), decoded["thresholds"][0].cpu().numpy(),
            decoded["weights"][0].cpu().numpy(), decoded["padding_logits"][0].cpu().numpy(),
            scorecard_id, target_n_rules=target_n_rules,
        )
        leaves = rows_to_leaves(rows, self.data.feature_names)
        prior = float(decoded["prior"][0].cpu().numpy() * self.corpus.prior_std + self.corpus.prior_mean)
        prior = float(np.clip(prior, self.corpus.prior_min, self.corpus.prior_max))
        _, _, scorecard_cls = import_rulecard(self.source_dir)
        scorecard = scorecard_from_leaves(leaves, prior, self.data.model_classes, self.config, scorecard_cls)
        return DecodedScoreCard(rows, leaves, prior, info, scorecard)

    @torch.no_grad()
    def decode_many(
        self,
        latent_vectors: np.ndarray,
        target_n_rules: int | None = None,
    ) -> list[DecodedScoreCard]:
        """Decodifica un'intera popolazione con una sola chiamata al modello."""
        vectors = np.asarray(latent_vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("latent_vectors deve avere shape (popolazione, latent_dim).")
        z_tensor = torch.as_tensor(vectors, dtype=torch.float32, device=self.device)
        decoded = self.model.decode(z_tensor)
        if "type_logits" not in decoded:
            return [self.decode(z, target_n_rules=target_n_rules) for z in vectors]

        first_id = self.next_scorecard_id
        self.next_scorecard_id += len(vectors)
        _, _, scorecard_cls = import_rulecard(self.source_dir)
        candidates: list[DecodedScoreCard] = []
        for position in range(len(vectors)):
            output_np = {
                key: value[position].cpu().numpy()
                for key, value in decoded.items()
                if key != "prior"
            }
            rows, leaves, prior, info = decode_tree_output_to_rows(
                output_np,
                float(decoded["prior"][position].cpu().numpy()),
                first_id + position,
                self.corpus,
                self.data,
            )
            leaves, removed_impossible = _drop_logically_impossible_leaves(leaves)
            info["n_logically_impossible_removed"] = removed_impossible
            info["target_n_rules"] = target_n_rules
            scorecard = scorecard_from_leaves(
                leaves, prior, self.data.model_classes, self.config, scorecard_cls
            )
            info["n_rules_prefix"] = int(len(scorecard.scorecard))
            candidates.append(DecodedScoreCard(rows, leaves, prior, info, scorecard))
        return candidates

    def verify_roundtrip(self) -> None:
        _, _, scorecard_cls = import_rulecard(self.source_dir)
        errors = []
        for identifier in self.corpus.card_ids:
            original = self.corpus.corpus_df.loc[self.corpus.corpus_df["scorecard_id"] == identifier]
            prior = float(self.corpus.cards_df.loc[self.corpus.cards_df["scorecard_id"] == identifier, "prior_log_odds"].iloc[0])
            original_sc = scorecard_from_leaves(original, prior, self.data.model_classes, self.config, scorecard_cls)
            rows = self.corpus.tree_rows_df.loc[self.corpus.tree_rows_df["scorecard_id"] == identifier]
            reconstructed = rows_to_leaves(rows, self.data.feature_names)
            reconstructed_sc = scorecard_from_leaves(reconstructed, prior, self.data.model_classes, self.config, scorecard_cls)
            _, pred_original = scorecard_outputs(original_sc, self.data.X_train)
            _, pred_reconstructed = scorecard_outputs(reconstructed_sc, self.data.X_train)
            errors.append(float((pred_original != pred_reconstructed).mean()))
        if max(errors) != 0.0:
            raise RuntimeError("Round-trip matrice → ScoreCard non predittivamente invertibile.")
        if os.environ.get("GENRULECARD_CLEAN_LOGS") != "1":
            print(f"Round-trip matrice → ScoreCard.predict: {self.corpus.n_sc}/{self.corpus.n_sc} identici")
