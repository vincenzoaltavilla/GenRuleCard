from __future__ import annotations

from pathlib import Path
import copy
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split

from .artifacts import CorpusBundle, GeneticResult, ProcessedDataset, RunPaths, VAEArtifacts
from .config import ExperimentConfig
from .run_manager import create_run_paths, save_json
from ..data.loaders import load_and_split_dataset
from ..genetic.search import (
    MinimumRuleTargetNotReached,
    performance_score,
    run_genetic_search,
)
from ..fairness import FAIRNESS_FEATURES, group_fairness, sensitive_values
from ..graphics.plots import (
    save_ga_plot,
    save_f1_complexity_weight_sweep_plots,
    save_phase_timing_plot,
    save_rulecard_visualization,
    save_vae_reconstruction_8plots,
    save_vae_training_plots,
)
from ..rulecard_fit.corpus import build_rulecard_corpus, select_best_initial_scorecard
from ..rulecard_fit.encoding import encode_corpus, scorecard_tensor_frame
from ..rulecard_fit.scorecards import import_rulecard, scorecard_from_leaves, scorecard_outputs
from ..utils.seeds import set_global_seed
from ..vae.decoder import ConstrainedDecoder
from ..vae.diagnostics import diagnose_reconstruction, diagnose_representation
from ..vae.model import build_vae
from ..vae.trainer import _structural_distance, encode_all, train_vae
from ..vae.tree_matrix import canonicalize_corpus_tree_order, tree_input_dim


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_label) -> float:
    y_true_pos = y_true == positive_label
    y_pred_pos = y_pred == positive_label
    tp = float(np.logical_and(y_true_pos, y_pred_pos).sum())
    fp = float(np.logical_and(~y_true_pos, y_pred_pos).sum())
    fn = float(np.logical_and(y_true_pos, ~y_pred_pos).sum())
    denominator = 2.0 * tp + fp + fn
    return 0.0 if denominator == 0.0 else float(2.0 * tp / denominator)


class GenRuleCard:
    """RuleCard → VAE → genetic search, con test isolato fino a predict()."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.run_paths: RunPaths | None = None
        self.data: ProcessedDataset | None = None
        self.corpus: CorpusBundle | None = None
        self.vae: VAEArtifacts | None = None
        self.decoder: ConstrainedDecoder | None = None
        self.best_initial = None
        self.genetic: GeneticResult | None = None
        self.final_report: pd.DataFrame | None = None
        self._is_fitted = False
        self._test_evaluated = False
        self.phase_timings: dict[str, float] = {}

    @property
    def run_dir(self) -> Path:
        if self.run_paths is None:
            raise RuntimeError("La pipeline non è stata ancora avviata.")
        return self.run_paths.root

    def _save_corpus(self, failures: list[dict]) -> None:
        assert self.corpus is not None and self.run_paths is not None
        c, out = self.corpus, self.run_paths.rulecard
        c.corpus_df.to_pickle(out / "initial_scorecards_leaves.pkl")
        c.corpus_df.to_csv(out / "initial_scorecards_leaves_readable.csv", index=False)
        c.cards_df.to_csv(out / "initial_scorecards_validation_metrics.csv", index=False)
        leaf_meta_cols = [
            column for column in [
                "scorecard_id", "variant_id", "variant_slot", "validation_accuracy", "validation_f1",
                "validation_selection_score", "validation_positive_prediction_rate", "sample_mode",
                "sample_size", "learning_rate", "top_k_features", "top_k_features_resolved", "reuse_features",
                "feature_fraction", "seed",
            ]
            if column in c.cards_df.columns
        ]
        if leaf_meta_cols:
            c.corpus_df.merge(c.cards_df[leaf_meta_cols], on="scorecard_id", how="left").to_csv(
                out / "initial_scorecards_leaves_with_metadata.csv", index=False
            )
        variant_cols = [
            "variant_id", "sample_mode", "learning_rate", "top_k_features",
            "reuse_features", "feature_fraction",
        ]
        if set(variant_cols).issubset(c.cards_df.columns):
            (
                c.cards_df
                .groupby(variant_cols)
                .agg(
                    n_scorecards=("scorecard_id", "size"),
                    validation_accuracy_mean=("validation_accuracy", "mean"),
                    validation_accuracy_std=("validation_accuracy", "std"),
                    validation_accuracy_min=("validation_accuracy", "min"),
                    validation_accuracy_q25=("validation_accuracy", lambda values: values.quantile(0.25)),
                    validation_accuracy_q50=("validation_accuracy", lambda values: values.quantile(0.50)),
                    validation_accuracy_q75=("validation_accuracy", lambda values: values.quantile(0.75)),
                    validation_accuracy_max=("validation_accuracy", "max"),
                    validation_f1_mean=("validation_f1", "mean"),
                    validation_f1_max=("validation_f1", "max"),
                    validation_selection_score_mean=("validation_selection_score", "mean"),
                    validation_selection_score_max=("validation_selection_score", "max"),
                    validation_positive_prediction_rate_mean=("validation_positive_prediction_rate", "mean"),
                    n_trees_mean=("n_trees", "mean"),
                    n_leaves_mean=("n_leaves", "mean"),
                )
                .reset_index()
                .sort_values(["validation_selection_score_mean", "n_scorecards"], ascending=[False, False])
                .to_csv(out / "variant_validation_summary.csv", index=False)
            )
        c.tree_rows_df.to_pickle(out / "canonical_scorecard_rows.pkl")
        c.tree_rows_df.to_csv(out / "canonical_scorecard_rows.csv", index=False)
        c.pattern_vocab.to_csv(out / "pattern_vocabulary.csv", index=False)
        scorecard_tensor_frame(c).to_csv(out / "vae_input_matrix_onehot.csv", index=False)
        np.save(out / "scorecard_tensor.npy", c.scorecard_tensor)
        np.savez_compressed(
            out / "scorecard_targets_and_masks.npz",
            pat_target=c.pat_target, thr_target=c.thr_target, weight_target=c.weight_target,
            pad_target=c.pad_target, has_child=c.has_child, real_mask=c.real_mask,
            thr1_mask=c.thr1_mask, thr2_mask=c.thr2_mask, weight_mask=c.weight_mask,
            prior_log_odds=c.prior_log_odds, prior_target=c.prior_target,
        )
        save_json({
            "n_scorecards": c.n_sc, "t_max": c.t_max, "n_rows": c.n_rows, "n_patterns": c.n_pat,
            "row_dim": c.row_dim, "weight_mean": c.weight_mean, "weight_std": c.weight_std,
            "prior_mean": c.prior_mean, "prior_std": c.prior_std, "prior_min": c.prior_min, "prior_max": c.prior_max,
            "best_initial_id": c.best_initial_id,
            "best_initial_validation_accuracy": c.best_initial_validation_accuracy,
            "encoding": "pattern_onehot",
        }, out / "normalization_and_corpus_metadata.json")
        if failures:
            pd.DataFrame(failures).to_csv(out / "scorecard_reconstruction_failures.csv", index=False)
        joblib.dump(c, out / "initial_scorecards_bundle.joblib", compress=3)

    def _load_latest_corpus(self) -> tuple[CorpusBundle | None, Path | None]:
        if not self.config.rulecard.reuse_latest_corpus:
            return None, None
        explicit_bundle = os.environ.get("GENRULECARD_CORPUS_BUNDLE")
        if explicit_bundle:
            path = Path(explicit_bundle).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Corpus esplicito non trovato: {path}"
                )
            corpus = joblib.load(path)
            is_onehot = (
                getattr(corpus, "row_dim", None)
                == getattr(corpus, "n_pat", -5) + 5
            )
            if getattr(corpus, "n_sc", None) != self.config.rulecard.n_cards or not is_onehot:
                raise RuntimeError(
                    "Corpus esplicito incompatibile: "
                    f"atteso={self.config.rulecard.n_cards}, "
                    f"trovato={getattr(corpus, 'n_sc', None)}, onehot={is_onehot}"
                )
            print(f"Corpus esplicito riusato: {path}", flush=True)
            return corpus, path
        root = Path(self.config.output_root)
        pattern = f"{self.config.dataset.name}_*_seed{self.config.seed}/01_rulecard_fit/initial_scorecards_bundle.joblib"
        # Cerca sia nella cartella campagna corrente sia nelle altre campagne
        # sotto la stessa output_root. Molte run benchmark hanno struttura:
        # experiments/benchmark_.../dataset_.../01_rulecard_fit/...
        search_roots = [root]
        # ``output_root`` può essere annidato (per esempio
        # experiments/18-09/nuova_campagna). Risali fino alla directory
        # experiments, invece di supportare soltanto campagne sue figlie
        # dirette.
        experiments_root = next(
            (parent for parent in (root, *root.parents) if parent.name == "experiments"),
            None,
        )
        if experiments_root is not None and experiments_root != root:
            search_roots.append(experiments_root)
        candidates = []
        for search_root in search_roots:
            candidates.extend(search_root.rglob(pattern))
        candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)
        rejection_counts: dict[str, int] = {
            "config_non_leggibile": 0,
            "split_diverso": 0,
            "rulecard_diversa": 0,
            "load_fallito": 0,
            "shape_non_compatibile": 0,
        }
        rejection_examples: list[str] = []

        def remember(reason: str, path: Path, detail: str = "") -> None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            if len(rejection_examples) < 3:
                suffix = f" ({detail})" if detail else ""
                rejection_examples.append(f"{reason}: {path.parent.parent.name}{suffix}")

        for path in candidates:
            config_path = path.parent.parent / "config.json"
            resolved_config_path = path.parent.parent / "resolved_config.json"
            try:
                source_config_path = resolved_config_path if resolved_config_path.exists() else config_path
                saved_config = json.loads(source_config_path.read_text(encoding="utf-8"))
            except Exception:
                remember("config_non_leggibile", path)
                continue
            if saved_config.get("split") != self.config.to_dict().get("split"):
                remember("split_diverso", path)
                continue
            saved_rulecard = dict(saved_config.get("rulecard", {}))
            current_rulecard = dict(self.config.to_dict().get("rulecard", {}))
            legacy_sample_modes = {
                "subsample_65": "majority_65",
                "subsample_80": "majority_80",
            }
            for rulecard in (saved_rulecard, current_rulecard):
                if "sample_modes" in rulecard:
                    rulecard["sample_modes"] = [
                        legacy_sample_modes.get(mode, mode)
                        for mode in rulecard["sample_modes"]
                    ]
            # Queste opzioni riguardano riuso/selezione, non la generazione
            # delle ScoreCard contenute nel corpus.
            selection_only = {
                "reuse_latest_corpus",
                "initial_selection_metric",
                "initial_selection_accuracy_weight",
                "s1_choice",
                "s1_f1_weight",
                "exclude_full_train_when_imbalanced",
                "full_train_minority_fraction_threshold",
                "reject_constant_validation_predictions",
                "exclude_constant_from_initial_selection",
            }
            for key in selection_only:
                saved_rulecard.pop(key, None)
                current_rulecard.pop(key, None)
            saved_rulecard = {
                key: saved_rulecard.get(key) for key in current_rulecard
            }
            if saved_rulecard != current_rulecard:
                diff_keys = [
                    key for key in current_rulecard
                    if saved_rulecard.get(key) != current_rulecard.get(key)
                ]
                detail = ", ".join(diff_keys[:3])
                remember("rulecard_diversa", path, detail=detail)
                continue
            try:
                corpus = joblib.load(path)
            except Exception:
                remember("load_fallito", path)
                continue
            is_onehot = getattr(corpus, "row_dim", None) == getattr(corpus, "n_pat", -5) + 5
            if getattr(corpus, "n_sc", None) == self.config.rulecard.n_cards and is_onehot:
                print(f"Corpus compatibile trovato: {path.parent.parent.name}")
                return corpus, path
            remember("shape_non_compatibile", path)
        loose_pattern = f"{self.config.dataset.name}_*/01_rulecard_fit/initial_scorecards_bundle.joblib"
        loose_paths: set[Path] = set()
        for search_root in search_roots:
            loose_paths.update(search_root.rglob(loose_pattern))
        loose_candidates = sorted(loose_paths, key=lambda path: path.stat().st_mtime, reverse=True)
        minimal_logs = os.environ.get("GENRULECARD_CLEAN_LOGS") == "1"
        if minimal_logs:
            print("Nessun corpus compatibile trovato")
        elif not candidates:
            print(
                "Nessun corpus compatibile trovato | "
                f"pattern={pattern} | candidati_stesso_seed=0 | "
                f"candidati_stesso_dataset_altri_seed={len(loose_candidates)}"
            )
        else:
            rejected = ", ".join(
                f"{reason}={count}" for reason, count in rejection_counts.items() if count
            ) or "nessun dettaglio"
            print(
                "Nessun corpus compatibile trovato | "
                f"candidati_stesso_seed={len(candidates)} | scartati: {rejected}"
            )
        return None, None

    @staticmethod
    def _corpus_fingerprint(corpus: CorpusBundle) -> str:
        """Identifica l'input effettivo del VAE, indipendentemente dalla cartella della run."""
        digest = hashlib.sha256()
        for value in (corpus.n_sc, corpus.n_rows, corpus.n_pat, corpus.row_dim):
            digest.update(str(int(value)).encode("ascii"))
            digest.update(b"|")
        digest.update(np.asarray(corpus.card_ids, dtype=np.int64).tobytes())
        for array in (corpus.scorecard_tensor, corpus.prior_target):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
        return digest.hexdigest()

    def _load_latest_vae(self):
        """Carica un VAE solo se corpus e configurazione risolta coincidono esattamente."""
        assert self.corpus is not None and self.run_paths is not None
        if not self.config.vae.reuse_latest_vae:
            return None
        root = Path(self.config.output_root)
        state_name = (
            "vae_earlystop_state.pt"
            if self.config.vae.deployment_checkpoint == "earlystop"
            else "vae_final_train_state.pt"
        )
        pattern = f"{self.config.dataset.name}_*_seed{self.config.seed}/02_vae/{state_name}"
        candidates = list(root.glob(pattern))
        experiments_root = next(
            (parent for parent in (root, *root.parents) if parent.name == "experiments"),
            None,
        )
        if experiments_root is not None:
            candidates.extend(experiments_root.rglob(pattern))
        candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)
        # Normalizza tuple/Path negli stessi tipi JSON usati dai config salvati.
        # Senza questo passaggio, ad esempio hidden_dims=(...) veniva confrontato
        # con hidden_dims=[...] e un checkpoint perfettamente compatibile era
        # scartato.
        current_vae = json.loads(json.dumps(
            self.config.to_dict().get("vae", {}), default=str
        ))
        current_vae.pop("reuse_latest_vae", None)
        current_vae.pop("initial_checkpoint", None)
        current_vae.pop("fine_tune_original_corpus_size", None)
        current_vae.setdefault("auto_pattern_weight_mode", "discrete")
        current_fingerprint = self._corpus_fingerprint(self.corpus)
        required = [
            "latent_embeddings.npy", "latent_distance_matrix.npy",
            "structural_distance_matrix.npy", "predictive_distance_matrix.npy",
            "representation_distance_diagnostics.csv", "latent_pca.csv",
            "vae_earlystop_history.csv", "vae_final_train_history.csv",
        ]
        for state_path in candidates:
            source_vae_dir = state_path.parent
            source_run = source_vae_dir.parent
            resolved_path = source_run / "resolved_config.json"
            config_path = resolved_path if resolved_path.exists() else source_run / "config.json"
            source_corpus_path = source_run / "01_rulecard_fit" / "initial_scorecards_bundle.joblib"
            if not config_path.exists() or not source_corpus_path.exists():
                continue
            if any(not (source_vae_dir / name).exists() for name in required):
                continue
            try:
                saved_config = json.loads(config_path.read_text(encoding="utf-8"))
                saved_vae = dict(saved_config.get("vae", {}))
                saved_vae.pop("reuse_latest_vae", None)
                saved_vae.pop("initial_checkpoint", None)
                saved_vae.pop("fine_tune_original_corpus_size", None)
                saved_vae.pop("n_starts", None)
                saved_vae.pop("start_seed_stride", None)
                saved_vae.setdefault("auto_pattern_weight_mode", "discrete")
                saved_vae.setdefault("deployment_checkpoint", "final_refit")
                saved_vae.setdefault("reconstruction_diagnostic_scope", "all")
                if saved_vae != current_vae:
                    continue
                source_corpus = joblib.load(source_corpus_path)
                if self._corpus_fingerprint(source_corpus) != current_fingerprint:
                    continue
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = build_vae(self.corpus, self.config.vae, device, self.data)
                try:
                    state = torch.load(state_path, map_location=device, weights_only=True)
                except TypeError:
                    state = torch.load(state_path, map_location=device)
                model.load_state_dict(state)
                model.eval()
                early_history = pd.read_csv(source_vae_dir / "vae_earlystop_history.csv")
                final_history = pd.read_csv(source_vae_dir / "vae_final_train_history.csv")
                report = pd.read_csv(source_vae_dir / "representation_distance_diagnostics.csv")
                pca = pd.read_csv(source_vae_dir / "latent_pca.csv")
                embeddings = np.load(source_vae_dir / "latent_embeddings.npy")
                structural = np.load(source_vae_dir / "structural_distance_matrix.npy")
                latent = np.load(source_vae_dir / "latent_distance_matrix.npy")
                predictive = np.load(source_vae_dir / "predictive_distance_matrix.npy")
                best_epoch = (
                    int(early_history.loc[early_history["vae_earlystop_monitored"].idxmin(), "epoch"])
                    if self.config.vae.deployment_checkpoint == "earlystop"
                    else int(final_history["epoch"].max())
                )
            except Exception:
                continue
            for source in source_vae_dir.iterdir():
                if source.is_file():
                    shutil.copy2(source, self.run_paths.vae / source.name)
            save_json({
                "source_run": source_run.name,
                "corpus_fingerprint": current_fingerprint,
                "compatibility": "exact corpus fingerprint and resolved VAE configuration",
            }, self.run_paths.vae / "reused_vae.json")
            return model, best_epoch, early_history, final_history, report, pca, structural, latent, predictive, embeddings
        return None

    def _load_notebook_vae(self):
        """Importa in modo tracciabile il checkpoint validato dal notebook."""
        assert self.corpus is not None and self.data is not None and self.run_paths is not None
        configured_root = self.config.vae.notebook_checkpoint_root
        if self.config.vae.vae_profile != "fmi_f1_sig" or not configured_root:
            return None
        source_dir = Path(configured_root) / str(self.config.dataset.name)
        checkpoint_path = source_dir / "model_VAE_fmi_f1_sig.pt"
        corpus_candidates = sorted(source_dir.glob("corpus_*.joblib"))
        if not checkpoint_path.exists() or not corpus_candidates:
            return None
        source_corpus = joblib.load(corpus_candidates[-1])
        if self._corpus_fingerprint(source_corpus) != self._corpus_fingerprint(self.corpus):
            print("Checkpoint notebook ignorato: fingerprint del corpus differente.")
            return None
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        expected = {
            "tree_input_dim": tree_input_dim(len(self.data.feature_names)),
            "t_max": int(self.corpus.t_max),
            "n_features": len(self.data.feature_names),
            "latent_dim": self.config.vae.latent_dim,
            "hidden": tuple(self.config.vae.hidden_dims or ()),
            "dropout": float(self.config.vae.dropout_p),
        }
        actual = {
            "tree_input_dim": checkpoint.get("tree_input_dim"),
            "t_max": checkpoint.get("t_max"),
            "n_features": checkpoint.get("n_features"),
            "latent_dim": checkpoint.get("latent_dim"),
            "hidden": tuple(checkpoint.get("hidden", ())),
            "dropout": float(checkpoint.get("dropout", np.nan)),
        }
        if actual != expected:
            print(f"Checkpoint notebook ignorato: metadati incompatibili | atteso={expected} | trovato={actual}")
            return None
        rename = {
            "mu.": "mu_head.", "logvar.": "logvar_head.",
            "type_logits.": "type_head.", "root_logits.": "root_head.",
            "left_child_logits.": "left_child_head.",
            "right_child_logits.": "right_child_head.",
            "thr.": "threshold_head.", "weight.": "weight_head.",
            "prior_out.": "prior_head.",
        }
        converted_state = {}
        for key, value in checkpoint["state_dict"].items():
            converted_key = key
            for old_prefix, new_prefix in rename.items():
                if key.startswith(old_prefix):
                    converted_key = new_prefix + key[len(old_prefix):]
                    break
            converted_state[converted_key] = value
        # Replica esatta del notebook: gli embedding vengono calcolati su
        # DEVICE (GPU se disponibile); soltanto dopo il modello passa su CPU
        # per il decoder del GA.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_vae(self.corpus, self.config.vae, device, self.data)
        model.load_state_dict(converted_state, strict=True)
        model.eval()

        history_path = source_dir / "history_VAE_fmi_f1_sig.csv"
        refit_path = source_dir / "refit_all_history_VAE_fmi_f1_sig.csv"
        source_history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        source_refit = pd.read_csv(refit_path) if refit_path.exists() else pd.DataFrame()
        early_history = pd.DataFrame({
            "epoch": source_history.get("epoch", pd.Series(dtype=float)),
            "vae_train_total": source_history.get("train_loss", pd.Series(dtype=float)),
            "vae_earlystop_total": source_history.get("holdout_loss", pd.Series(dtype=float)),
        })
        if not source_history.empty:
            early_history["vae_earlystop_monitored"] = (
                source_history["holdout_feature"]
                + source_history["holdout_signature_count"]
                + source_history["holdout_count_leaves"]
                + 0.5 * source_history["holdout_f1_loss"]
                + 0.5 * source_history["holdout_metric"]
            )
        final_history = pd.DataFrame({
            "epoch": source_refit.get("epoch", pd.Series(dtype=float)),
            "full_T_total": source_refit.get("loss", pd.Series(dtype=float)),
        })
        early_history.to_csv(self.run_paths.vae / "vae_earlystop_history.csv", index=False)
        final_history.to_csv(self.run_paths.vae / "vae_final_train_history.csv", index=False)
        torch.save(model.state_dict(), self.run_paths.vae / "vae_final_train_state.pt")

        # Il checkpoint e il corpus arrivano dallo stesso benchmark: importa
        # anche il relativo audit VAE a otto pannelli e i dati che lo generano.
        notebook_report = source_dir / (
            f"report_{self.config.dataset.name}_VAE_fmi_f1_sig_natural_8plots.png"
        )
        if notebook_report.exists():
            shutil.copy2(
                notebook_report,
                self.run_paths.figures / "vae_reconstruction_fmi_8plots.png",
            )
        for diagnostic_name in (
            "decoded_reconstruction_VAE_fmi_f1_sig.csv",
            "decoded_failures_VAE_fmi_f1_sig.csv",
            "summary_VAE_fmi_f1_sig.csv",
        ):
            diagnostic_source = source_dir / diagnostic_name
            if diagnostic_source.exists():
                shutil.copy2(diagnostic_source, self.run_paths.vae / diagnostic_name)

        distance_candidates = sorted(source_dir.glob("D_fmi_*.npy"))
        structural = (
            np.load(distance_candidates[-1]).astype(np.float32)
            if distance_candidates else _structural_distance(self.corpus, self.config.vae, self.data)
        )
        embeddings = encode_all(model, self.corpus, self.config.vae.batch_size, self.data)
        model = model.cpu().eval()
        report, pca, latent, predictive = diagnose_representation(
            embeddings, structural, self.corpus, self.data, self.config.rulecard,
            self.config.rulecard_source_dir, self.run_paths.vae, self.config.seed,
        )
        best_epoch = (
            int(source_history.loc[early_history["vae_earlystop_monitored"].idxmin(), "epoch"])
            if not source_history.empty else 0
        )
        save_json({
            "source_checkpoint": str(checkpoint_path.resolve()),
            "source_corpus": str(corpus_candidates[-1].resolve()),
            "corpus_fingerprint": self._corpus_fingerprint(self.corpus),
            "compatibility": "exact corpus fingerprint, architecture metadata and strict state_dict",
        }, self.run_paths.vae / "imported_notebook_vae.json")
        print(f"VAE importato dal notebook | dataset={self.config.dataset.name} | checkpoint={checkpoint_path}")
        return model, best_epoch, early_history, final_history, report, pca, structural, latent, predictive, embeddings

    def _resolve_vae_profile(self) -> dict:
        """Risolve il solo peso pattern usando statistiche del corpus, mai il test."""
        assert self.corpus is not None and self.run_paths is not None
        c = self.corpus
        if getattr(self.config.vae, "representation", "row") == "tree":
            report = {
                "vae_profile": self.config.vae.vae_profile,
                "selected_profile": "tree_matrix",
                "n_scorecards": int(c.n_sc),
                "n_patterns": int(c.n_pat),
                "pattern_to_scorecard_ratio": float(c.n_pat / max(c.n_sc, 1)),
                "requested_pattern_loss_weight": self.config.vae.loss_weights.get("pattern"),
                "auto_pattern_weight_mode": self.config.vae.auto_pattern_weight_mode,
                "resolved_pattern_loss_weight": self.config.vae.loss_weights.get("pattern"),
                "reason": "TreeVAE usa una matrice ad alberi; il profilo pattern del vecchio VAE non viene applicato.",
            }
            save_json(report, self.run_paths.vae / "vae_profile_resolution.json")
            save_json(self.config.to_dict(), self.run_paths.root / "resolved_config.json")
            if os.environ.get("GENRULECARD_CLEAN_LOGS") != "1":
                print(
                    "VAE profile: "
                    "representation=tree | selected=tree_matrix | "
                    "pattern profile bypassed"
                )
            return report
        counts = c.pattern_vocab["count"].to_numpy(dtype=np.int64)
        total_occurrences = int(counts.sum())
        rare_frequency_threshold = max(2, math.ceil(0.005 * c.n_sc))
        rare_mask = counts <= rare_frequency_threshold
        rare_pattern_types = int(rare_mask.sum())
        rare_pattern_type_fraction = rare_pattern_types / max(c.n_pat, 1)
        rare_pattern_occurrences = int(counts[rare_mask].sum())
        rare_pattern_mass = rare_pattern_occurrences / max(total_occurrences, 1)
        pattern_ratio = c.n_pat / max(c.n_sc, 1)

        requested_weight = float(self.config.vae.loss_weights["pattern"])
        if self.config.vae.vae_profile == "auto":
            ratio_excess = max(0.0, (pattern_ratio - 0.35) / 0.40)
            rare_mass_excess = max(0.0, (rare_pattern_mass - 0.01) / 0.005)
            if self.config.vae.auto_pattern_weight_mode == "discrete":
                ratio_component = None
                rare_mass_component = None
                if pattern_ratio >= 0.75 or rare_pattern_mass >= 0.015:
                    selected_profile = "high_pattern_complexity"
                    resolved_weight = 6.0
                    reasons = []
                    if pattern_ratio >= 0.75:
                        reasons.append("N_PAT/N_SC >= 0.75")
                    if rare_pattern_mass >= 0.015:
                        reasons.append("rare_pattern_mass >= 0.015")
                    reason = "; ".join(reasons)
                elif pattern_ratio < 0.35 and rare_pattern_mass < 0.01:
                    selected_profile = "low_pattern_complexity"
                    resolved_weight = 3.0
                    reason = "N_PAT/N_SC < 0.35 and rare_pattern_mass < 0.01"
                else:
                    selected_profile = "standard"
                    resolved_weight = 4.5
                    reason = "intermediate pattern complexity"
            elif self.config.vae.auto_pattern_weight_mode == "smooth_soft":
                ratio_component = 2.5 * math.log1p(ratio_excess)
                rare_mass_component = 1.5 * math.log1p(rare_mass_excess)
                resolved_weight = 2.0 + ratio_component + rare_mass_component
                selected_profile = "smooth_soft_pattern_complexity"
                reason = (
                    "2 + 2.5*log1p(max(0,(N_PAT/N_SC-0.35)/0.40)) "
                    "+ 1.5*log1p(max(0,(rare_pattern_mass-0.01)/0.005))"
                )
            else:
                ratio_component = 3.0 * math.log1p(ratio_excess)
                rare_mass_component = 2.0 * math.log1p(rare_mass_excess)
                resolved_weight = 3.0 + ratio_component + rare_mass_component
                selected_profile = "smooth_aggressive_pattern_complexity"
                reason = (
                    "3 + 3*log1p(max(0,(N_PAT/N_SC-0.35)/0.40)) "
                    "+ 2*log1p(max(0,(rare_pattern_mass-0.01)/0.005))"
                )
            self.config.vae.loss_weights = dict(self.config.vae.loss_weights)
            self.config.vae.loss_weights["pattern"] = resolved_weight
        else:
            selected_profile = "standard"
            resolved_weight = requested_weight
            reason = "vae_profile='standard': configured loss_weights preserved"

        report = {
            "vae_profile": self.config.vae.vae_profile,
            "selected_profile": selected_profile,
            "n_scorecards": int(c.n_sc),
            "n_patterns": int(c.n_pat),
            "pattern_to_scorecard_ratio": float(pattern_ratio),
            "total_pattern_occurrences": total_occurrences,
            "rare_frequency_threshold": rare_frequency_threshold,
            "rare_pattern_types": rare_pattern_types,
            "rare_pattern_type_fraction": float(rare_pattern_type_fraction),
            "rare_pattern_occurrences": rare_pattern_occurrences,
            "rare_pattern_mass": float(rare_pattern_mass),
            "requested_pattern_loss_weight": requested_weight,
            "auto_pattern_weight_mode": self.config.vae.auto_pattern_weight_mode,
            "pattern_ratio_excess": float(ratio_excess) if self.config.vae.vae_profile == "auto" else None,
            "rare_pattern_mass_excess": float(rare_mass_excess) if self.config.vae.vae_profile == "auto" else None,
            "pattern_ratio_component": (
                float(ratio_component) if ratio_component is not None else None
            ),
            "rare_pattern_mass_component": (
                float(rare_mass_component) if rare_mass_component is not None else None
            ),
            "resolved_pattern_loss_weight": float(resolved_weight),
            "reason": reason,
        }
        save_json(report, self.run_paths.vae / "vae_profile_resolution.json")
        save_json(self.config.to_dict(), self.run_paths.root / "resolved_config.json")
        print(
            "VAE profile: "
            f"mode={report['vae_profile']} | selected={selected_profile} | "
            f"N_PAT/N_SC={pattern_ratio:.3f} | rare_mass={rare_pattern_mass:.3%} | "
            f"pattern_weight={resolved_weight:.1f} | reason={reason}"
        )
        return report

    def _tensor_from_rows(self, rows: pd.DataFrame, prior_log_odds: float) -> dict[str, np.ndarray]:
        assert self.corpus is not None and self.data is not None
        c, data = self.corpus, self.data
        pat_target = np.full((1, c.n_rows), -100, dtype=np.int64)
        thr_target = np.zeros((1, c.n_rows, 2), dtype=np.float32)
        weight_target = np.zeros((1, c.n_rows, 2), dtype=np.float32)
        pad_target = np.ones((1, c.n_rows), dtype=np.float32)
        has_child = np.zeros((1, c.n_rows), dtype=np.float32)
        pat_onehot = np.zeros((1, c.n_rows, c.n_pat), dtype=np.float32)

        for row in rows.itertuples(index=False):
            if int(row.is_padding) == 1:
                continue
            row_pos = 2 * int(row.tree_idx) + int(row.row_in_tree)
            pattern_id = int(row.pattern_id)
            root_feat = int(row.root_feat)
            child_feat = int(row.child_feat)
            pat_target[0, row_pos] = pattern_id
            pat_onehot[0, row_pos, pattern_id] = 1.0
            pad_target[0, row_pos] = 0.0
            thr_target[0, row_pos, 0] = np.clip(
                (float(row.thr1) - data.feature_min[root_feat]) / data.feature_range[root_feat], 0.0, 1.0
            )
            if child_feat >= 0:
                has_child[0, row_pos] = 1.0
                thr_target[0, row_pos, 1] = np.clip(
                    (float(row.thr2) - data.feature_min[child_feat]) / data.feature_range[child_feat], 0.0, 1.0
                )
            weight_target[0, row_pos, 0] = (float(row.w) - c.weight_mean) / c.weight_std
            weight_target[0, row_pos, 1] = (float(row.w_counter) - c.weight_mean) / c.weight_std

        real_mask = 1.0 - pad_target
        tensor = np.concatenate(
            [pat_onehot, thr_target, weight_target, pad_target[..., None]], axis=-1
        ).astype(np.float32)
        prior = np.asarray([prior_log_odds], dtype=np.float32)
        prior_target = ((prior - c.prior_mean) / c.prior_std).astype(np.float32)
        return {
            "scorecard_tensor": tensor,
            "pat_target": pat_target,
            "thr_target": thr_target,
            "weight_target": weight_target,
            "pad_target": pad_target,
            "has_child": has_child,
            "real_mask": real_mask,
            "thr1_mask": real_mask,
            "thr2_mask": real_mask * has_child,
            "weight_mask": real_mask,
            "prior_log_odds": prior,
            "prior_target": prior_target,
        }

    def _genetic_comparison_marker(self) -> dict[str, float | str]:
        assert self.corpus is not None and self.data is not None and self.vae is not None
        assert self.genetic is not None and self.best_initial is not None
        c = self.corpus
        initial_id = int(self.best_initial["id"])
        initial_pos = c.card_ids.index(initial_id)
        genetic_id = max(c.card_ids) + 1
        genetic_rows = self.genetic.rows.copy()
        genetic_rows["scorecard_id"] = genetic_id
        encoded = self._tensor_from_rows(genetic_rows, self.genetic.prior_log_odds)
        augmented = CorpusBundle(
            corpus_df=c.corpus_df,
            cards_df=c.cards_df,
            tree_rows_df=pd.concat([c.tree_rows_df, genetic_rows], ignore_index=True),
            pattern_vocab=c.pattern_vocab,
            card_ids=c.card_ids + [genetic_id],
            scorecard_tensor=np.concatenate([c.scorecard_tensor, encoded["scorecard_tensor"]], axis=0),
            pat_target=np.concatenate([c.pat_target, encoded["pat_target"]], axis=0),
            thr_target=np.concatenate([c.thr_target, encoded["thr_target"]], axis=0),
            weight_target=np.concatenate([c.weight_target, encoded["weight_target"]], axis=0),
            pad_target=np.concatenate([c.pad_target, encoded["pad_target"]], axis=0),
            has_child=np.concatenate([c.has_child, encoded["has_child"]], axis=0),
            real_mask=np.concatenate([c.real_mask, encoded["real_mask"]], axis=0),
            thr1_mask=np.concatenate([c.thr1_mask, encoded["thr1_mask"]], axis=0),
            thr2_mask=np.concatenate([c.thr2_mask, encoded["thr2_mask"]], axis=0),
            weight_mask=np.concatenate([c.weight_mask, encoded["weight_mask"]], axis=0),
            prior_log_odds=np.concatenate([c.prior_log_odds, encoded["prior_log_odds"]], axis=0),
            prior_target=np.concatenate([c.prior_target, encoded["prior_target"]], axis=0),
            n_sc=c.n_sc + 1,
            t_max=c.t_max,
            n_rows=c.n_rows,
            n_pat=c.n_pat,
            row_dim=c.row_dim,
            weight_mean=c.weight_mean,
            weight_std=c.weight_std,
            prior_mean=c.prior_mean,
            prior_std=c.prior_std,
            prior_min=c.prior_min,
            prior_max=c.prior_max,
            best_initial_id=c.best_initial_id,
            best_initial_validation_accuracy=c.best_initial_validation_accuracy,
        )
        structural_augmented = _structural_distance(augmented, self.config.vae, self.data)
        initial_prob, _ = scorecard_outputs(self.best_initial["scorecard"], self.data.X_train)
        genetic_prob, _ = scorecard_outputs(self.genetic.scorecard, self.data.X_train)
        initial_prob = np.asarray(initial_prob, dtype=np.float32)
        genetic_prob = np.asarray(genetic_prob, dtype=np.float32)
        if initial_prob.ndim == 2:
            initial_prob = initial_prob[:, -1]
        if genetic_prob.ndim == 2:
            genetic_prob = genetic_prob[:, -1]
        predictive_distance = float(np.linalg.norm(initial_prob - genetic_prob) / np.sqrt(len(initial_prob)))
        latent_distance = float(np.linalg.norm(self.vae.embeddings[initial_pos] - self.genetic.best_z))
        return {
            "label": "best initial -> best genetic",
            "structural_distance": float(structural_augmented[initial_pos, -1]),
            "predictive_distance": predictive_distance,
            "latent_distance": latent_distance,
        }

    def fit(self, *, stop_after_vae: bool = False):
        if self._is_fitted:
            raise RuntimeError("Questo oggetto è già stato addestrato. Crea una nuova istanza per una nuova run.")
        set_global_seed(self.config.seed)
        self.run_paths = create_run_paths(self.config.output_root, self.config.dataset.name, self.config.seed)
        save_json(self.config.to_dict(), self.run_paths.root / "config.json")
        save_json({
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }, self.run_paths.root / "environment.json")
        minimal_logs = os.environ.get("GENRULECARD_CLEAN_LOGS") == "1"
        if not minimal_logs:
            print("=" * 72)
            print("GenRuleCard")
            print("=" * 72)
        self.data = load_and_split_dataset(self.config.dataset, self.config.split)
        if not minimal_logs:
            print(f"Dataset={self.data.name} | Train={self.data.X_train.shape} | Validation={self.data.X_validation.shape} | Test={self.data.X_test.shape}")
        balance_parts = []
        for split_name, labels in (
            ("train", self.data.y_train),
            ("validation", self.data.y_validation),
            ("test", self.data.y_test),
        ):
            n_total = len(labels)
            n_positive = int(np.sum(labels == self.data.model_classes[-1]))
            n_negative = n_total - n_positive
            balance_parts.append(
                f"{split_name}: class0={n_negative} ({n_negative / n_total:.1%}), "
                f"class1={n_positive} ({n_positive / n_total:.1%})"
            )
        if not minimal_logs:
            print("Bilanciamento dataset | " + " | ".join(balance_parts))
        save_json(self.config.to_dict(), self.run_paths.root / "resolved_config.json")
        phase_started = time.perf_counter()
        print("\n[1/4] Generazione corpus ScoreCard")
        notebook_corpus_path = None
        if self.config.vae.vae_profile == "fmi_f1_sig" and self.config.vae.notebook_checkpoint_root:
            notebook_dir = Path(self.config.vae.notebook_checkpoint_root) / str(self.config.dataset.name)
            notebook_candidates = sorted(notebook_dir.glob("corpus_*.joblib"))
            if notebook_candidates:
                notebook_corpus_path = notebook_candidates[-1]
        if notebook_corpus_path is not None:
            reusable_corpus, reusable_path = joblib.load(notebook_corpus_path), notebook_corpus_path
            print(f"Corpus importato dal benchmark notebook: {notebook_corpus_path}")
        else:
            reusable_corpus, reusable_path = self._load_latest_corpus()
        if reusable_corpus is not None:
            self.corpus = reusable_corpus
            corpus_df, cards_df, failures = self.corpus.corpus_df, self.corpus.cards_df, []
            print("Corpus compatibile riusato da una campagna precedente.")
            initial_id, initial_val, initial_leaves, initial_scorecard = select_best_initial_scorecard(
                corpus_df, cards_df, self.data, self.config.rulecard,
                self.config.rulecard_source_dir, self.config.genetic, self.config.seed,
            )
            self.corpus.best_initial_id = initial_id
            self.corpus.best_initial_validation_accuracy = initial_val
            self._save_corpus(failures)
        else:
            corpus_df, cards_df, failures = build_rulecard_corpus(
                self.data, self.config.rulecard, self.config.seed, self.config.rulecard_source_dir
            )
            initial_id, initial_val, initial_leaves, initial_scorecard = select_best_initial_scorecard(
                corpus_df, cards_df, self.data, self.config.rulecard,
                self.config.rulecard_source_dir, self.config.genetic, self.config.seed,
            )
            self.corpus = encode_corpus(corpus_df, cards_df, self.data, initial_id, initial_val)
            self._save_corpus(failures)
        self.best_initial = {
            "id": initial_id, "validation_accuracy": initial_val,
            "leaves": initial_leaves, "scorecard": initial_scorecard,
            "prior_log_odds": float(cards_df.loc[cards_df["scorecard_id"] == initial_id, "prior_log_odds"].iloc[0]),
        }
        assert self.corpus is not None
        if self.config.vae.vae_profile == "fmi_f1_sig":
            self.corpus = canonicalize_corpus_tree_order(self.corpus)
            if not minimal_logs:
                print("Tree order canonicalizzato per il profilo VAE_fmi_f1_sig.")
        initial_leaves.to_csv(self.run_paths.rulecard / "best_initial_scorecard_leaves.csv", index=False)
        self.corpus.tree_rows_df.loc[self.corpus.tree_rows_df["scorecard_id"] == initial_id].to_csv(
            self.run_paths.rulecard / "best_initial_scorecard_matrix.csv", index=False
        )
        initial_meta = cards_df.loc[cards_df["scorecard_id"] == initial_id].iloc[0].to_dict()
        _, initial_validation_prediction = scorecard_outputs(
            initial_scorecard, self.data.X_validation
        )
        initial_validation_f1 = _binary_f1(
            self.data.y_validation,
            initial_validation_prediction,
            self.data.model_classes[-1],
        )
        _, initial_test_prediction = scorecard_outputs(
            initial_scorecard, self.data.X_test
        )
        initial_test_f1 = _binary_f1(
            self.data.y_test,
            initial_test_prediction,
            self.data.model_classes[-1],
        )
        absolute_mode = (
            self.config.genetic.f1_complexity_rule_objective == "absolute_min_rules"
        )
        if not absolute_mode:
            print(
            "Best ScoreCard iniziale selezionata su validation\n"
            f"  validation | F1={initial_validation_f1:.4f} | "
            f"errori={int(np.sum(initial_validation_prediction != self.data.y_validation))}/"
            f"{len(self.data.y_validation)}\n"
            f"  test       | F1={initial_test_f1:.4f} | "
            f"errori={int(np.sum(initial_test_prediction != self.data.y_test))}/"
            f"{len(self.data.y_test)}\n"
            f"  n_rules={int(len(initial_scorecard.scorecard))}"
            )
        else:
            pass
        variant_keys = [
            "variant_id", "variant_slot", "sample_mode", "sample_size", "learning_rate",
            "top_k_features", "reuse_features", "feature_fraction", "selected_features",
            "n_selected_features", "seed",
        ]
        initial_variant = {key: initial_meta.get(key) for key in variant_keys if key in initial_meta}
        save_json(initial_variant, self.run_paths.rulecard / "best_initial_scorecard_variant.json")
        if not absolute_mode:
            print("Variante migliore iniziale: " + " | ".join(
                f"{key}={value}" for key, value in initial_variant.items()
                if key not in {"variant_id", "variant_slot", "selected_features"}
            ))
        if getattr(self.config.vae, "representation", "row") == "tree":
            print(
                f"Encoding: N_SC={self.corpus.n_sc} | "
                f"T_MAX={self.corpus.t_max} | N_ROWS={self.corpus.n_rows}"
            )
            if not minimal_logs:
                print(
                    "VAE representation | "
                    f"tree-level | one row per tree | N_FEATURES={len(self.data.feature_names)}"
                )
        else:
            print(
                f"Encoding: N_SC={self.corpus.n_sc} | T_MAX={self.corpus.t_max} | "
                f"N_ROWS={self.corpus.n_rows} | N_PAT={self.corpus.n_pat}"
            )
        self._resolve_vae_profile()
        self.phase_timings["phase_1_rulecard_corpus"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        print("\n[2/4] Training VAE")
        vae_config = self.config.vae
        # Il checkpoint validato dal notebook ha precedenza sui VAE riaddestrati
        # dal main; il fallback resta il normale riuso di una campagna ufficiale.
        reused_vae = self._load_notebook_vae()
        if reused_vae is None:
            reused_vae = self._load_latest_vae()
        if reused_vae is not None:
            (
                model, best_epoch, early_history, final_history, report, pca,
                structural_distance, latent_distance, predictive_distance, embeddings,
            ) = reused_vae
            reused_info = json.loads(
                (self.run_paths.vae / "reused_vae.json").read_text(encoding="utf-8")
            )
            metric_weight = float(self.config.vae.loss_weights.get("metric", 0.0))
            loss_label = (
                "strutturale+0.005*KL"
                if metric_weight == 0.0 else
                f"strutturale+0.005*KL+{metric_weight:g}*FMI"
            )
            print(f"VAE riusato | loss={loss_label} | source={reused_info['source_run']}")
        else:
            model, best_epoch, early_history, final_history, structural_distance = train_vae(
                self.corpus, vae_config, self.config.seed, self.run_paths.vae,
                self.data, self.config.rulecard, self.config.rulecard_source_dir,
            )
            embeddings = encode_all(model, self.corpus, self.config.vae.batch_size, self.data)
            representation_indices = None
            if self.config.vae.reconstruction_diagnostic_scope == "holdout":
                diagnostic_population = np.arange(self.corpus.n_sc)
                if (
                    self.config.vae.initial_checkpoint is not None
                    and self.config.vae.fine_tune_original_corpus_size is not None
                ):
                    diagnostic_population = np.arange(
                        self.config.vae.fine_tune_original_corpus_size
                    )
                _, representation_indices = train_test_split(
                    diagnostic_population, test_size=0.20,
                    random_state=self.config.seed,
                )
            report, pca, latent_distance, predictive_distance = diagnose_representation(
                embeddings, structural_distance, self.corpus, self.data, self.config.rulecard,
                self.config.rulecard_source_dir, self.run_paths.vae, self.config.seed,
                diagnostic_indices=representation_indices,
            )
        save_vae_training_plots(early_history, final_history, self.run_paths.figures)
        self.vae = VAEArtifacts(
            model=model, embeddings=embeddings, best_epoch=best_epoch, earlystop_history=early_history,
            final_history=final_history, representation_report=report, latent_pca=pca,
            structural_distance=structural_distance, latent_distance=latent_distance, predictive_distance=predictive_distance,
        )
        self.decoder = ConstrainedDecoder(
            model, self.corpus, self.data, self.config.rulecard,
            self.config.rulecard_source_dir, self.config.vae,
        )
        self.decoder.verify_roundtrip()
        diagnostic_embeddings = embeddings
        diagnostic_corpus = self.corpus
        diagnostic_structural = structural_distance
        diagnostic_latent = latent_distance
        if self.config.vae.reconstruction_diagnostic_scope == "holdout":
            _, holdout_idx = train_test_split(
                np.arange(self.corpus.n_sc), test_size=0.20,
                random_state=self.config.seed,
            )
            holdout_idx = np.asarray(holdout_idx, dtype=np.int64)
            diagnostic_embeddings = embeddings[holdout_idx]
            diagnostic_structural = structural_distance[np.ix_(holdout_idx, holdout_idx)]
            diagnostic_latent = latent_distance[np.ix_(holdout_idx, holdout_idx)]
            diagnostic_corpus = copy.copy(self.corpus)
            diagnostic_corpus.card_ids = [self.corpus.card_ids[int(i)] for i in holdout_idx]
            save_json({
                "scope": "holdout",
                "seed": self.config.seed,
                "n_scorecards": len(holdout_idx),
                "indices": holdout_idx.tolist(),
                "scorecard_ids": diagnostic_corpus.card_ids,
            }, self.run_paths.vae / "reconstruction_diagnostic_scope.json")
            if not minimal_logs:
                print(f"Diagnostica VAE a 8 pannelli: solo holdout 20% ({len(holdout_idx)} ScoreCard).")
        reconstruction_report = diagnose_reconstruction(
            diagnostic_embeddings, self.decoder, diagnostic_corpus,
            self.data, self.run_paths.vae
        )
        save_vae_reconstruction_8plots(
            reconstruction_report,
            diagnostic_structural,
            diagnostic_latent,
            (
                f"{self.config.dataset.name} | holdout 20%"
                if self.config.vae.reconstruction_diagnostic_scope == "holdout"
                else self.config.dataset.name
            ),
            self.run_paths.figures,
        )

        self.phase_timings["phase_2_vae"] = time.perf_counter() - phase_started
        if stop_after_vae:
            print("\nFit VAE completato: ricerca genetica non eseguita.")
            return self

        phase_started = time.perf_counter()
        print("\n[3/4] Ricerca genetica")
        if self.config.vae.vae_profile == "fmi_f1_sig":
            # Stesso preflight del notebook, incluso l'avanzamento dell'id
            # interno del decoder prima della popolazione GA.
            preflight = self.decoder.decode(np.asarray(embeddings[0], dtype=np.float32))
            _, preflight_prediction = scorecard_outputs(
                preflight.scorecard, self.data.X_validation
            )
            preflight_f1 = _binary_f1(
                self.data.y_validation,
                preflight_prediction,
                self.data.model_classes[-1],
            )
            if not minimal_logs:
                print(
                f"GA preflight | valid=True | f1={preflight_f1:.4f} | "
                f"leaves={len(preflight.leaves)} | "
                f"rules={len(preflight.scorecard.scorecard)}"
                )
        ga_seed = (
            self.config.seed
            if self.config.genetic.random_seed is None
            else int(self.config.genetic.random_seed)
        )
        self.genetic = run_genetic_search(
            embeddings, self.decoder, self.data, self.config.genetic, ga_seed,
            initial_val, self.run_paths.genetic,
            best_initial_scorecard=self.best_initial["scorecard"],
        )
        save_ga_plot(self.genetic.history, self.run_paths.figures)
        comparison_marker = None
        try:
            comparison_marker = self._genetic_comparison_marker()
            save_json(comparison_marker, self.run_paths.genetic / "initial_to_genetic_distance_marker.json")
        except Exception as error:
            save_json({
                "status": "skipped",
                "reason": repr(error),
            }, self.run_paths.genetic / "initial_to_genetic_distance_marker.json")

        # Stesso plot del notebook: regole leggibili + punteggio rosso/verde.
        visual = self.config.visualization
        figure_kwargs = dict(
            onehot_mapping=self.data.metadata.get("onehot_mapping", {}),
            boolean_features=self.data.metadata.get("boolean_features", []),
            max_rules=visual.scorecard_max_rules,
            figsize=visual.scorecard_figsize,
            row_height=visual.scorecard_row_height,
            show=visual.show_scorecard_windows,
        )
        display_classes = self.data.metadata.get("display_classes", self.data.model_classes)
        initial_readable_rules = save_rulecard_visualization(
            self.best_initial["scorecard"],
            self.data.feature_names,
            display_classes,
            self.run_paths.figures / "best_initial_scorecard.png",
            **figure_kwargs,
        )
        genetic_readable_rules = save_rulecard_visualization(
            self.genetic.scorecard,
            self.data.feature_names,
            display_classes,
            self.run_paths.figures / "best_genetic_scorecard.png",
            **figure_kwargs,
        )
        initial_readable_rules.to_csv(
            self.run_paths.rulecard / "best_initial_scorecard_readable_rules.csv", index=False
        )
        genetic_readable_rules.to_csv(
            self.run_paths.genetic / "best_genetic_scorecard_readable_rules.csv", index=False
        )
        save_json({
            "validation_accuracy": self.genetic.validation_accuracy,
            "validation_f1": self.genetic.info.get("validation_f1"),
            "selection_fitness": self.genetic.info.get("selection_fitness"),
            "fold_performance_mean": self.genetic.info.get("fold_performance_mean"),
            "fold_performance_std": self.genetic.info.get("fold_performance_std"),
            "fold_performance_min": self.genetic.info.get("fold_performance_min"),
            "robust_validation_performance": self.genetic.info.get("robust_validation_performance"),
            "confidence_tiebreak": self.genetic.info.get("confidence_tiebreak"),
            "complexity_penalty": self.genetic.info.get("complexity_penalty"),
            "reference_n_scorecard_rules": self.genetic.info.get("reference_n_scorecard_rules"),
            "redundancy_penalty": self.genetic.info.get("redundancy_penalty"),
            "prior_log_odds": self.genetic.prior_log_odds,
            "n_active_trees": self.genetic.info.get("n_active"),
        }, self.run_paths.genetic / "best_candidate_metadata.json")
        self.phase_timings["phase_3_genetic_search"] = time.perf_counter() - phase_started
        self._is_fitted = True
        if not minimal_logs:
            print("\n[4/4] Fit completato. La ScoreCard genetica non è ancora stata valutata sul test.")
        return self

    def run_f1_complexity_weight_sweep(
        self,
        weights: list[float] | tuple[float, ...],
        *,
        fairness_context: dict | None = None,
        sweep_subdir: str = "03_genetic_weight_sweep",
    ) -> pd.DataFrame:
        """Rilancia solo il GA sul VAE già addestrato, variando il peso F1."""
        if (
            not self._is_fitted or self.data is None or self.corpus is None
            or self.vae is None or self.best_initial is None or self.genetic is None
            or self.run_paths is None
        ):
            raise RuntimeError("Prima esegui fit().")
        weights = sorted({round(float(value), 10) for value in weights})
        if any(value < 0.0 or value > 1.0 for value in weights):
            raise ValueError("I pesi F1 devono essere compresi tra 0 e 1.")
        sweep_root = self.run_paths.root / sweep_subdir
        sweep_root.mkdir(parents=True, exist_ok=True)
        positive_label = self.data.model_classes[-1]
        _, initial_validation_prediction = scorecard_outputs(
            self.best_initial["scorecard"], self.data.X_validation
        )
        _, initial_test_prediction = scorecard_outputs(
            self.best_initial["scorecard"], self.data.X_test
        )
        initial_validation_f1 = _binary_f1(
            self.data.y_validation, initial_validation_prediction, positive_label
        )
        initial_test_f1 = _binary_f1(
            self.data.y_test, initial_test_prediction, positive_label
        )
        initial_n_rules = int(len(self.best_initial["scorecard"].scorecard))
        configured_weight = round(
            float(self.config.genetic.f1_complexity_f1_weight), 10
        )
        absolute_mode = (
            self.config.genetic.f1_complexity_rule_objective == "absolute_min_rules"
        )
        rows = []
        rows_by_target: dict[int, dict] = {}
        resume_source_run = None
        reused_vae_path = self.run_paths.vae / "reused_vae.json"
        if reused_vae_path.exists():
            try:
                reused_vae_info = json.loads(reused_vae_path.read_text(encoding="utf-8"))
                candidate_run = self.run_paths.root.parent / str(reused_vae_info["source_run"])
                if candidate_run.exists() and candidate_run != self.run_paths.root:
                    resume_source_run = candidate_run
            except Exception:
                resume_source_run = None

        def load_completed_result(weight: float) -> tuple[GeneticResult, Path] | None:
            """Recupera un alpha concluso da una run interrotta con lo stesso VAE."""
            if resume_source_run is None:
                return None
            source = (
                resume_source_run / "03_genetic"
                if abs(weight) < 1e-12 else
                resume_source_run / "03_genetic_weight_sweep" / f"f1_weight_{weight:.1f}"
            )
            required = [
                source / "best_genetic_latent_vector.npy",
                source / "best_genetic_fitness_metrics.csv",
                source / "genetic_history.csv",
                source / "best_genetic_scorecard_leaves.csv",
            ]
            if any(not path.exists() for path in required):
                return None
            vector = np.load(source / "best_genetic_latent_vector.npy").reshape(-1)
            candidate = self.decoder.decode(np.asarray(vector, dtype=np.float32))
            saved_metrics = pd.read_csv(source / "best_genetic_fitness_metrics.csv").iloc[0].to_dict()
            saved_rules = int(float(saved_metrics["n_scorecard_rules"]))
            if len(candidate.scorecard.scorecard) != saved_rules:
                return None
            result = GeneticResult(
                best_z=np.asarray(vector, dtype=np.float32),
                validation_accuracy=float(saved_metrics["validation_accuracy"]),
                rows=candidate.rows,
                leaves=candidate.leaves,
                prior_log_odds=float(candidate.prior_log_odds),
                scorecard=candidate.scorecard,
                history=pd.read_csv(source / "genetic_history.csv"),
                info=saved_metrics,
            )
            return result, source

        def target_for(weight: float) -> int:
            return max(1, int(np.floor(weight * initial_n_rules + 0.5)))

        def append_result(weight: float, result: GeneticResult, output_dir: Path) -> None:
            _, test_prediction = scorecard_outputs(result.scorecard, self.data.X_test)
            test_f1 = _binary_f1(self.data.y_test, test_prediction, positive_label)
            genetic_n_rules = int(len(result.scorecard.scorecard))
            target_n_rules = np.nan if absolute_mode else target_for(weight)
            row = {
                "f1_weight": float(weight),
                "complexity_weight": float(1.0 - weight),
                "requested_rule_reduction": np.nan if absolute_mode else float(1.0 - weight),
                "rule_constraint_active": False if absolute_mode else bool(weight < 1.0 - 1e-12),
                "target_n_rules": target_n_rules,
                "actual_target_reduction": np.nan if absolute_mode else float(
                    (initial_n_rules - target_n_rules) / initial_n_rules
                ),
                "initial_validation_f1": initial_validation_f1,
                "genetic_validation_f1": float(result.info["validation_f1"]),
                "delta_validation_f1": float(
                    result.info["validation_f1"] - initial_validation_f1
                ),
                "initial_test_f1": initial_test_f1,
                "genetic_test_f1": test_f1,
                "delta_test_f1": float(test_f1 - initial_test_f1),
                "initial_n_rules": initial_n_rules,
                "genetic_n_rules": genetic_n_rules,
                "delta_n_rules": genetic_n_rules - initial_n_rules,
                "best_fitness": float(result.info["selection_fitness"]),
                "status": "ok",
                "reused_from_f1_weight": np.nan,
                "output_dir": str(output_dir),
            }
            if fairness_context is not None:
                validation_metrics = group_fairness(
                    self.data.y_validation,
                    scorecard_outputs(result.scorecard, self.data.X_validation)[1],
                    fairness_context["sensitive_validation"],
                )
                test_metrics = group_fairness(
                    self.data.y_test, test_prediction,
                    fairness_context["sensitive_test"],
                )
                row.update({
                    "fairness_objective": fairness_context["feature"],
                    "fairness_reference_median_dpd": fairness_context["reference"]["median_dpd"],
                    "fairness_reference_median_eodd": fairness_context["reference"]["median_eodd"],
                    "validation_dpd": validation_metrics["demographic_parity_difference"],
                    "validation_equal_opportunity": validation_metrics["equal_opportunity_difference"],
                    "validation_equalized_odds": validation_metrics["equalized_odds_difference"],
                    "test_dpd": test_metrics["demographic_parity_difference"],
                    "test_equal_opportunity": test_metrics["equal_opportunity_difference"],
                    "test_equalized_odds": test_metrics["equalized_odds_difference"],
                    "fairness_penalty": result.info.get("fairness_penalty"),
                    "fairness_component": result.info.get("fairness_component"),
                })
                for audit_feature, audit_values in fairness_context.get("audit", {}).items():
                    audit_validation = group_fairness(
                        self.data.y_validation,
                        scorecard_outputs(result.scorecard, self.data.X_validation)[1],
                        audit_values["validation"],
                    )
                    audit_test = group_fairness(
                        self.data.y_test, test_prediction, audit_values["test"],
                    )
                    prefix = f"audit_{audit_feature}"
                    row.update({
                        f"{prefix}_validation_dpd": audit_validation["demographic_parity_difference"],
                        f"{prefix}_validation_equal_opportunity": audit_validation["equal_opportunity_difference"],
                        f"{prefix}_validation_equalized_odds": audit_validation["equalized_odds_difference"],
                        f"{prefix}_test_dpd": audit_test["demographic_parity_difference"],
                        f"{prefix}_test_equal_opportunity": audit_test["equal_opportunity_difference"],
                        f"{prefix}_test_equalized_odds": audit_test["equalized_odds_difference"],
                    })
            rows.append(row)
            # Ogni alpha e' un esperimento GA completo: salva anche le due
            # figure e la versione leggibile della ScoreCard nella sua cartella.
            visual = self.config.visualization
            display_classes = self.data.metadata.get(
                "display_classes", self.data.model_classes
            )
            readable = save_rulecard_visualization(
                result.scorecard,
                self.data.feature_names,
                display_classes,
                output_dir / "best_genetic_scorecard.png",
                onehot_mapping=self.data.metadata.get("onehot_mapping", {}),
                boolean_features=self.data.metadata.get("boolean_features", []),
                max_rules=visual.scorecard_max_rules,
                figsize=visual.scorecard_figsize,
                row_height=visual.scorecard_row_height,
                show=False,
            )
            readable.to_csv(
                output_dir / "best_genetic_scorecard_readable_rules.csv", index=False
            )
            save_ga_plot(result.history, output_dir)

        def append_unreached_target(weight: float, output_dir: Path, error: Exception) -> None:
            target_n_rules = target_for(weight)
            rows.append({
                "f1_weight": float(weight),
                "complexity_weight": float(1.0 - weight),
                "requested_rule_reduction": float(1.0 - weight),
                "rule_constraint_active": True,
                "target_n_rules": target_n_rules,
                "actual_target_reduction": float(
                    (initial_n_rules - target_n_rules) / initial_n_rules
                ),
                "initial_validation_f1": initial_validation_f1,
                "genetic_validation_f1": np.nan,
                "delta_validation_f1": np.nan,
                "initial_test_f1": initial_test_f1,
                "genetic_test_f1": np.nan,
                "delta_test_f1": np.nan,
                "initial_n_rules": initial_n_rules,
                "genetic_n_rules": np.nan,
                "delta_n_rules": np.nan,
                "best_fitness": np.nan,
                "status": "minimum_target_not_reached",
                "reused_from_f1_weight": np.nan,
                "error": str(error),
                "output_dir": str(output_dir),
            })

        def append_reused(weight: float, source: dict) -> None:
            target_n_rules = target_for(weight)
            replicated = dict(source)
            replicated.update({
                "f1_weight": float(weight),
                "complexity_weight": float(1.0 - weight),
                "requested_rule_reduction": float(1.0 - weight),
                "rule_constraint_active": bool(weight < 1.0 - 1e-12),
                "target_n_rules": target_n_rules,
                "actual_target_reduction": float(
                    (initial_n_rules - target_n_rules) / initial_n_rules
                ),
                "reused_from_f1_weight": float(source["f1_weight"]),
            })
            rows.append(replicated)
            print(
                f"GA weight sweep | F1={weight:.1f} | target={target_n_rules} "
                f"già calcolato con F1={source['f1_weight']:.1f}: risultato replicato"
            )

        ga_seed = (
            self.config.seed
            if self.config.genetic.random_seed is None
            else int(self.config.genetic.random_seed)
        )
        for weight in weights:
            target_n_rules = target_for(weight)
            constraint_active = (not absolute_mode) and weight < 1.0 - 1e-12
            # weight=1 è il protocollo F1 puro: il vincolo sulle regole è
            # disattivato, quindi non è equivalente a un peso <1 che produce
            # numericamente lo stesso target.
            if constraint_active and target_n_rules in rows_by_target:
                append_reused(weight, rows_by_target[target_n_rules])
                continue
            if fairness_context is None and weight == configured_weight:
                append_result(weight, self.genetic, self.run_paths.genetic)
                if constraint_active:
                    rows_by_target[target_n_rules] = rows[-1]
                continue
            completed = load_completed_result(weight)
            if completed is not None:
                result, source_dir = completed
                output_dir = sweep_root / f"f1_weight_{weight:.1f}"
                shutil.copytree(source_dir, output_dir, dirs_exist_ok=True)
                append_result(weight, result, output_dir)
                print(f"GA resume | alpha={weight:.1f} riusato dalla run interrotta")
                continue
            if os.environ.get("GENRULECARD_CLEAN_LOGS") == "1":
                print()
            if os.environ.get("GENRULECARD_CLEAN_LOGS") != "1":
                print("\n" + "-" * 72)
                print(
                    f"GA weight sweep | F1={weight:.1f} | "
                    f"complessità={1.0 - weight:.1f}"
                )
            ga_config = copy.deepcopy(self.config.genetic)
            ga_config.f1_complexity_f1_weight = weight
            ga_config.f1_complexity_rule_objective = (
                "absolute_min_rules" if absolute_mode else "target_constraint"
            )
            output_dir = sweep_root / f"f1_weight_{weight:.1f}"
            decoder = ConstrainedDecoder(
                self.vae.model, self.corpus, self.data, self.config.rulecard,
                self.config.rulecard_source_dir, self.config.vae,
            )
            # Replica lo stesso avanzamento del decoder usato nel fit standard.
            if self.config.vae.vae_profile == "fmi_f1_sig":
                decoder.decode(np.asarray(self.vae.embeddings[0], dtype=np.float32))
            try:
                result = run_genetic_search(
                    self.vae.embeddings, decoder, self.data, ga_config, ga_seed,
                    self.best_initial["validation_accuracy"], output_dir,
                    best_initial_scorecard=(
                        None if absolute_mode else self.best_initial["scorecard"]
                    ),
                    fairness_context=fairness_context,
                )
                append_result(weight, result, output_dir)
            except MinimumRuleTargetNotReached as error:
                print(f"GA weight sweep | {error}")
                append_unreached_target(weight, output_dir, error)
            if constraint_active:
                rows_by_target[target_n_rules] = rows[-1]

        results = pd.DataFrame(rows).sort_values("f1_weight").reset_index(drop=True)
        results.to_csv(sweep_root / "f1_complexity_weight_sweep.csv", index=False)
        save_f1_complexity_weight_sweep_plots(
            results, self.run_paths.figures, dataset=str(self.config.dataset.name)
        )
        if os.environ.get("GENRULECARD_CLEAN_LOGS") == "1" and not results.empty:
            final_row = results.iloc[-1]
            print(
                "GA early stopping | "
                f"best val_f1={final_row['genetic_validation_f1']:.4f} | "
                f"fitness={final_row['best_fitness']:.4f}"
            )
        summary_columns = (
            ["f1_weight", "genetic_validation_f1", "genetic_n_rules", "best_fitness"]
            if absolute_mode else
            ["f1_weight", "target_n_rules", "actual_target_reduction",
             "delta_validation_f1", "delta_test_f1", "genetic_n_rules", "delta_n_rules"]
        )
        if os.environ.get("GENRULECARD_CLEAN_LOGS") != "1":
            print("\nRiepilogo sweep F1-complexity")
            print(results[summary_columns].to_string(index=False))
        return results

    def run_fairness_weight_sweeps(
        self,
        weights: list[float] | tuple[float, ...],
        penalty_weight: float = 0.25,
    ) -> pd.DataFrame:
        """Sweep GA separati per attributo, usando le mediane fairness del corpus."""
        if self.data is None or self.corpus is None or self.run_paths is None:
            raise RuntimeError("Prima esegui fit().")
        dataset = str(self.config.dataset.name)
        features = FAIRNESS_FEATURES.get(dataset, ())
        if not features:
            raise ValueError(f"Nessun attributo sensibile configurato per {dataset}.")
        if penalty_weight < 0.0:
            raise ValueError("penalty_weight deve essere non negativo.")

        output_root = self.run_paths.root / "03_genetic_fairness"
        output_root.mkdir(parents=True, exist_ok=True)
        _, _, scorecard_cls = import_rulecard(self.config.rulecard_source_dir)
        cards_by_id = self.corpus.cards_df.set_index("scorecard_id")
        validation_sensitive = {
            feature: sensitive_values(dataset, self.data.raw_split["validation"], feature)
            for feature in features
        }
        test_sensitive = {
            feature: sensitive_values(dataset, self.data.raw_split["test"], feature)
            for feature in features
        }
        distribution_rows = []
        for identifier in self.corpus.card_ids:
            identifier = int(identifier)
            leaves = self.corpus.corpus_df.loc[
                self.corpus.corpus_df["scorecard_id"] == identifier
            ].copy()
            scorecard = scorecard_from_leaves(
                leaves, float(cards_by_id.loc[identifier, "prior_log_odds"]),
                self.data.model_classes, self.config.rulecard, scorecard_cls,
            )
            _, prediction = scorecard_outputs(scorecard, self.data.X_validation)
            if np.unique(prediction).size < 2:
                continue
            for feature in features:
                metrics = group_fairness(
                    self.data.y_validation, prediction, validation_sensitive[feature]
                )
                distribution_rows.append({
                    "scorecard_id": identifier,
                    "sensitive_feature": feature,
                    "validation_f1": float(cards_by_id.loc[identifier, "validation_f1"]),
                    "n_leaves": int(cards_by_id.loc[identifier, "n_leaves"]),
                    **metrics,
                })
        distributions = pd.DataFrame(distribution_rows)
        distributions.to_csv(output_root / "corpus_fairness_distributions.csv", index=False)
        metric_columns = [
            "demographic_parity_difference", "equal_opportunity_difference",
            "equalized_odds_difference",
        ]
        summary_rows = []
        for feature, feature_frame in distributions.groupby("sensitive_feature", sort=False):
            for metric in metric_columns:
                values = feature_frame[metric].dropna()
                summary_rows.append({
                    "sensitive_feature": feature,
                    "metric": metric,
                    "n_scorecards": int(len(values)),
                    "min": float(values.min()),
                    "q1": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "q3": float(values.quantile(0.75)),
                    "max": float(values.max()),
                })
        distribution_summary = pd.DataFrame(summary_rows)
        distribution_summary.to_csv(output_root / "corpus_fairness_summary.csv", index=False)

        audit = {
            feature: {
                "validation": validation_sensitive[feature],
                "test": test_sensitive[feature],
            }
            for feature in features
        }
        all_results = []
        for feature in features:
            feature_summary = distribution_summary.set_index(["sensitive_feature", "metric"])
            reference = {
                "median_dpd": float(feature_summary.loc[
                    (feature, "demographic_parity_difference"), "median"
                ]),
                "median_eodd": float(feature_summary.loc[
                    (feature, "equalized_odds_difference"), "median"
                ]),
            }
            context = {
                "feature": feature,
                "sensitive_validation": validation_sensitive[feature],
                "sensitive_test": test_sensitive[feature],
                "reference": reference,
                "penalty_weight": float(penalty_weight),
                "audit": audit,
            }
            print(
                f"\nFAIRNESS GA | dataset={dataset} | objective={feature} | "
                f"median_dpd={reference['median_dpd']:.4f} | "
                f"median_eodd={reference['median_eodd']:.4f} | lambda={penalty_weight:.3f}"
            )
            result = self.run_f1_complexity_weight_sweep(
                weights,
                fairness_context=context,
                sweep_subdir=f"03_genetic_fairness/{feature}",
            )
            if "fairness_objective" not in result.columns:
                result.insert(0, "fairness_objective", feature)
            all_results.append(result)
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(output_root / "fairness_weight_sweep.csv", index=False)
        return combined

    def predict(self) -> pd.DataFrame:
        """Valuta una sola volta i modelli selezionati sul test set interno."""
        if not self._is_fitted or self.data is None or self.run_paths is None or self.genetic is None or self.best_initial is None:
            raise RuntimeError("Prima esegui fit().")
        if self._test_evaluated and self.final_report is not None:
            return self.final_report.copy()
        phase_started = time.perf_counter()
        _, initial_pred = scorecard_outputs(self.best_initial["scorecard"], self.data.X_test)
        _, genetic_pred = scorecard_outputs(self.genetic.scorecard, self.data.X_test)
        _, initial_val_pred = scorecard_outputs(self.best_initial["scorecard"], self.data.X_validation)
        _, genetic_val_pred = scorecard_outputs(self.genetic.scorecard, self.data.X_validation)
        initial_acc = float((initial_pred == self.data.y_test).mean())
        genetic_acc = float((genetic_pred == self.data.y_test).mean())
        positive_label = self.data.model_classes[-1]
        initial_val_f1 = _binary_f1(self.data.y_validation, initial_val_pred, positive_label)
        genetic_val_f1 = float(self.genetic.info.get("validation_f1", np.nan))
        initial_test_f1 = _binary_f1(self.data.y_test, initial_pred, positive_label)
        genetic_test_f1 = _binary_f1(self.data.y_test, genetic_pred, positive_label)
        genetic_config = self.config.genetic
        initial_selection_score = performance_score(
            self.best_initial["validation_accuracy"], initial_val_f1,
            genetic_config.fitness_metric, genetic_config.accuracy_f1_accuracy_weight,
        )
        genetic_selection_score = performance_score(
            self.genetic.validation_accuracy, genetic_val_f1,
            genetic_config.fitness_metric, genetic_config.accuracy_f1_accuracy_weight,
        )
        self.final_report = pd.DataFrame([
            {
                "candidate": "best_initial_corpus", "validation_accuracy": self.best_initial["validation_accuracy"],
                "validation_f1": initial_val_f1,
                "test_accuracy": initial_acc,
                "test_f1": initial_test_f1,
                "validation_selection_score": initial_selection_score,
                "n_trees": int(self.best_initial["leaves"]["tree_idx"].nunique()),
                "n_leaves": int(len(self.best_initial["leaves"])),
                "n_scorecard_rules": int(len(self.best_initial["scorecard"].scorecard)),
            },
            {
                "candidate": "best_genetic", "validation_accuracy": self.genetic.validation_accuracy,
                "validation_f1": genetic_val_f1,
                "test_accuracy": genetic_acc,
                "test_f1": genetic_test_f1,
                "validation_selection_score": genetic_selection_score,
                "n_trees": int(self.genetic.leaves["tree_idx"].nunique()),
                "n_leaves": int(len(self.genetic.leaves)),
                "n_scorecard_rules": int(len(self.genetic.scorecard.scorecard)),
            },
        ])
        self.final_report.to_csv(self.run_paths.results / "final_comparison.csv", index=False)
        pd.DataFrame({
            "y_true": self.data.y_test,
            "best_initial_prediction": initial_pred,
            "best_genetic_prediction": genetic_pred,
        }).to_csv(self.run_paths.results / "test_predictions.csv", index=False)
        save_json({
            "protocol": "test set evaluated only in GenRuleCard.predict()",
            "best_initial_test_accuracy": initial_acc,
            "best_initial_test_f1": initial_test_f1,
            "best_genetic_test_accuracy": genetic_acc,
            "best_genetic_test_f1": genetic_test_f1,
            "selection_protocol": "no final candidate recommendation; initial and genetic scorecards are reported separately",
            "best_initial_validation_selection_score": initial_selection_score,
            "best_genetic_validation_selection_score": genetic_selection_score,
        }, self.run_paths.results / "final_summary.json")
        self.phase_timings["phase_4_final_evaluation"] = time.perf_counter() - phase_started
        phase_labels = {
            "phase_1_rulecard_corpus": "1. Corpus RuleCard",
            "phase_2_vae": "2. VAE e diagnostiche",
            "phase_3_genetic_search": "3. Ricerca genetica",
            "phase_4_final_evaluation": "4. Valutazione finale",
        }
        timing_frame = pd.DataFrame([
            {
                "phase": phase,
                "phase_label": phase_labels[phase],
                "seconds": seconds,
                "minutes": seconds / 60.0,
            }
            for phase, seconds in self.phase_timings.items()
        ])
        timing_frame.to_csv(self.run_paths.results / "phase_timings.csv", index=False)
        save_json(
            {**self.phase_timings, "total_seconds": float(sum(self.phase_timings.values()))},
            self.run_paths.results / "phase_timings.json",
        )
        save_phase_timing_plot(timing_frame, self.run_paths.figures)
        for phase, seconds in self.phase_timings.items():
            self.final_report[f"{phase}_seconds"] = seconds
        self.final_report.to_csv(self.run_paths.results / "final_comparison.csv", index=False)
        self._test_evaluated = True
        if os.environ.get("GENRULECARD_CLEAN_LOGS") != "1":
            print(
            "\nBest ScoreCard genetica selezionata su validation\n"
            f"  validation | F1={genetic_val_f1:.4f} | "
            f"errori={int(np.sum(genetic_val_pred != self.data.y_validation))}/"
            f"{len(self.data.y_validation)}\n"
            f"  test       | F1={genetic_test_f1:.4f} | "
            f"errori={int(np.sum(genetic_pred != self.data.y_test))}/"
            f"{len(self.data.y_test)}\n"
            f"  n_rules={int(len(self.genetic.scorecard.scorecard))}\n"
            "\nConfronto finale F1\n"
            f"  validation | iniziale={initial_val_f1:.4f} | "
            f"genetica={genetic_val_f1:.4f} | "
            f"delta={genetic_val_f1 - initial_val_f1:+.4f}\n"
            f"  test       | iniziale={initial_test_f1:.4f} | "
            f"genetica={genetic_test_f1:.4f} | "
            f"delta={genetic_test_f1 - initial_test_f1:+.4f}\n"
            f"  n_rules    | iniziale={int(len(self.best_initial['scorecard'].scorecard))} | "
            f"genetica={int(len(self.genetic.scorecard.scorecard))} | "
            f"delta={int(len(self.genetic.scorecard.scorecard) - len(self.best_initial['scorecard'].scorecard)):+d}"
            )
        return self.final_report.copy()
