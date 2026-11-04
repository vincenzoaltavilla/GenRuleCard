from __future__ import annotations

from collections import Counter
import re
import numpy as np
import pandas as pd

from ..core.artifacts import CorpusBundle, ProcessedDataset


def _column_safe_pattern_name(name: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_").lower()
    return clean or "pattern"


def scorecard_tensor_frame(corpus: CorpusBundle) -> pd.DataFrame:
    """Return the exact VAE matrix representation with readable column names."""
    vocab = corpus.pattern_vocab.sort_values("pattern_id").reset_index(drop=True)
    pattern_columns = [
        f"pattern_{int(row.pattern_id):03d}_{_column_safe_pattern_name(row.pattern_name)}"
        for row in vocab.itertuples(index=False)
    ]
    columns = pattern_columns + ["thr1", "thr2", "w", "w_counter", "is_padding"]
    flat_tensor = corpus.scorecard_tensor.reshape(corpus.n_sc * corpus.n_rows, corpus.row_dim)
    return pd.DataFrame(flat_tensor, columns=columns)


def branch_side(operator: str) -> str:
    return "L" if operator in ("<", "<=") else "R"


def padding_row(scorecard_id: int, tree_idx: int, row_in_tree: int) -> dict:
    return {
        "scorecard_id": int(scorecard_id), "tree_idx": int(tree_idx), "row_in_tree": int(row_in_tree),
        "kind": "PAD", "pattern_key": None, "pattern_name": "<PAD>",
        "root_feat": -1, "child_feat": -1, "thr1": 0.0, "thr2": 0.0,
        "w": 0.0, "w_counter": 0.0, "is_padding": 1,
    }


def make_real_row(
    scorecard_id: int, tree_idx: int, row_in_tree: int, kind: str,
    root_feat: int, child_feat: int, thr1: float, thr2: float,
    w: float, w_counter: float, feature_names: list[str],
) -> dict:
    pattern_key = ("stump", int(root_feat)) if int(child_feat) < 0 else ("pair", int(root_feat), int(child_feat))
    pattern_name = feature_names[int(root_feat)] if int(child_feat) < 0 else f"{feature_names[int(root_feat)]} → {feature_names[int(child_feat)]}"
    return {
        "scorecard_id": int(scorecard_id), "tree_idx": int(tree_idx), "row_in_tree": int(row_in_tree),
        "kind": kind, "pattern_key": pattern_key, "pattern_name": pattern_name,
        "root_feat": int(root_feat), "child_feat": int(child_feat),
        "thr1": float(thr1), "thr2": float(thr2), "w": float(w),
        "w_counter": float(w_counter), "is_padding": 0,
    }


def tree_to_two_rows(tree_df: pd.DataFrame, scorecard_id: int, feature_names: list[str]) -> list[dict]:
    tree_idx = int(tree_df["tree_idx"].iloc[0])
    n_leaves = len(tree_df)
    if n_leaves == 2:
        paths = list(tree_df["conditions_key"])
        if not all(len(path) == 1 for path in paths):
            raise ValueError("Stump non valido.")
        signatures = {(int(path[0][0]), float(path[0][1])) for path in paths}
        if len(signatures) != 1:
            raise ValueError("Soglia radice incoerente.")
        root_feat, root_thr = signatures.pop()
        weights = {branch_side(row["conditions_key"][0][2]): float(row["w"]) for _, row in tree_df.iterrows()}
        if set(weights) != {"L", "R"}:
            raise ValueError("Foglie stump non complementari.")
        return [
            make_real_row(scorecard_id, tree_idx, 0, "STUMP", root_feat, -1, root_thr, 0.0, weights["L"], weights["R"], feature_names),
            padding_row(scorecard_id, tree_idx, 1),
        ]
    if n_leaves not in (3, 4):
        raise ValueError(f"Numero foglie non supportato: {n_leaves}")
    paths = list(tree_df["conditions_key"])
    if not all(1 <= len(path) <= 2 for path in paths):
        raise ValueError("Tree con più di due condizioni.")
    signatures = {(int(path[0][0]), float(path[0][1])) for path in paths}
    if len(signatures) != 1:
        raise ValueError("Soglia radice incoerente.")
    root_feat, root_thr = signatures.pop()
    rows: list[dict] = []
    for row_in_tree, side in enumerate(("L", "R")):
        branch_leaves = [row for _, row in tree_df.iterrows() if branch_side(row["conditions_key"][0][2]) == side]
        if len(branch_leaves) == 1:
            leaf = branch_leaves[0]
            if len(leaf["conditions_key"]) != 1:
                raise ValueError("Ramo diretto non valido.")
            weight = float(leaf["w"])
            rows.append(make_real_row(scorecard_id, tree_idx, row_in_tree, f"{side}_LEAF", root_feat, -1, root_thr, 0.0, weight, weight, feature_names))
        elif len(branch_leaves) == 2:
            child_signatures = {(int(row["conditions_key"][1][0]), float(row["conditions_key"][1][1])) for row in branch_leaves}
            if len(child_signatures) != 1:
                raise ValueError("Soglia figlio incoerente.")
            child_feat, child_thr = child_signatures.pop()
            weights = {branch_side(row["conditions_key"][1][2]): float(row["w"]) for row in branch_leaves}
            if set(weights) != {"L", "R"}:
                raise ValueError("Foglie figlio non complementari.")
            rows.append(make_real_row(scorecard_id, tree_idx, row_in_tree, side, root_feat, child_feat, root_thr, child_thr, weights["L"], weights["R"], feature_names))
        else:
            raise ValueError("Struttura di ramo non valida.")
    return rows


def rows_to_leaves(rows_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    leaves: list[dict] = []
    for tree_idx, tree_df in rows_df.groupby("tree_idx", sort=True):
        real = tree_df.query("is_padding == 0").sort_values("row_in_tree").reset_index(drop=True)
        if len(real) == 0:
            continue
        if len(real) == 1:
            row = real.iloc[0]
            leaves.extend([
                {"tree_idx": int(tree_idx), "conditions_key": ((int(row["root_feat"]), float(row["thr1"]), "<="),), "w": float(row["w"]), "support": 0},
                {"tree_idx": int(tree_idx), "conditions_key": ((int(row["root_feat"]), float(row["thr1"]), ">"),), "w": float(row["w_counter"]), "support": 0},
            ])
        elif len(real) == 2:
            for _, row in real.iterrows():
                root_op = "<=" if int(row["row_in_tree"]) == 0 else ">"
                root = (int(row["root_feat"]), float(row["thr1"]), root_op)
                if int(row["child_feat"]) < 0:
                    leaves.append({"tree_idx": int(tree_idx), "conditions_key": (root,), "w": float(row["w"]), "support": 0})
                else:
                    left = (int(row["child_feat"]), float(row["thr2"]), "<=")
                    right = (int(row["child_feat"]), float(row["thr2"]), ">")
                    leaves.extend([
                        {"tree_idx": int(tree_idx), "conditions_key": (root, left), "w": float(row["w"]), "support": 0},
                        {"tree_idx": int(tree_idx), "conditions_key": (root, right), "w": float(row["w_counter"]), "support": 0},
                    ])
        else:
            raise ValueError("Tree canonico con numero di righe reali non valido.")
    leaves_df = pd.DataFrame(leaves)
    if leaves_df.empty:
        raise ValueError("Scorecard senza foglie.")
    leaves_df["rule"] = leaves_df["conditions_key"].apply(
        lambda path: " AND ".join(f"{feature_names[int(f)]} {op} {float(t):.8g}" for f, t, op in path)
    )
    return leaves_df.sort_values(["tree_idx", "rule"]).reset_index(drop=True)


def encode_corpus(
    corpus_df: pd.DataFrame,
    cards_df: pd.DataFrame,
    data: ProcessedDataset,
    best_initial_id: int,
    best_initial_validation_accuracy: float,
) -> CorpusBundle:
    canonical_rows: list[dict] = []
    failures: list[tuple] = []
    for (scorecard_id, _), tree_df in corpus_df.groupby(["scorecard_id", "tree_idx"], sort=True):
        try:
            canonical_rows.extend(tree_to_two_rows(tree_df, int(scorecard_id), data.feature_names))
        except Exception as error:
            failures.append((int(scorecard_id), int(tree_df["tree_idx"].iloc[0]), str(error)))
    if failures:
        raise RuntimeError(f"Errore encoding canonico: {failures[:3]}")
    tree_rows = pd.DataFrame(canonical_rows).sort_values(["scorecard_id", "tree_idx", "row_in_tree"]).reset_index(drop=True)
    real_rows = tree_rows.query("is_padding == 0").copy()
    vocab = (
        real_rows[["pattern_key", "pattern_name", "root_feat", "child_feat"]]
        .drop_duplicates("pattern_key")
        .sort_values(["pattern_name", "root_feat", "child_feat"])
        .reset_index(drop=True)
    )
    vocab.insert(0, "pattern_id", np.arange(len(vocab)))
    vocab["count"] = vocab["pattern_key"].map(Counter(real_rows["pattern_key"].tolist()))
    pattern_to_id = dict(zip(vocab["pattern_key"], vocab["pattern_id"]))
    tree_rows["pattern_id"] = tree_rows["pattern_key"].apply(lambda key: pattern_to_id.get(key, -1)).astype(int)
    card_ids = sorted(cards_df["scorecard_id"].tolist())
    card_to_pos = {identifier: position for position, identifier in enumerate(card_ids)}
    n_sc = len(card_ids)
    t_max = int(corpus_df.groupby("scorecard_id")["tree_idx"].nunique().max())
    n_rows = 2 * t_max
    n_pat = len(vocab)
    weights = tree_rows.loc[tree_rows["is_padding"] == 0, ["w", "w_counter"]].to_numpy(np.float32)
    w_mean = float(weights.mean())
    w_std = float(weights.std()) or 1.0
    pat_target = np.full((n_sc, n_rows), -100, dtype=np.int64)
    thr_target = np.zeros((n_sc, n_rows, 2), dtype=np.float32)
    weight_target = np.zeros((n_sc, n_rows, 2), dtype=np.float32)
    pad_target = np.ones((n_sc, n_rows), dtype=np.float32)
    has_child = np.zeros((n_sc, n_rows), dtype=np.float32)
    pat_onehot = np.zeros((n_sc, n_rows, n_pat), dtype=np.float32)
    real = tree_rows.query("is_padding == 0").copy()
    real["card_pos"] = real["scorecard_id"].map(card_to_pos)
    real["row_pos"] = 2 * real["tree_idx"] + real["row_in_tree"]
    card_pos = real["card_pos"].to_numpy()
    row_pos = real["row_pos"].to_numpy()
    pattern_id = real["pattern_id"].to_numpy()
    root_feat = real["root_feat"].to_numpy()
    child_feat = real["child_feat"].to_numpy()
    thr1 = real["thr1"].to_numpy(np.float32)
    thr2 = real["thr2"].to_numpy(np.float32)
    w = real["w"].to_numpy(np.float32)
    w_counter = real["w_counter"].to_numpy(np.float32)
    pat_onehot[card_pos, row_pos, pattern_id] = 1.0
    pat_target[card_pos, row_pos] = pattern_id
    pad_target[card_pos, row_pos] = 0.0
    thr_target[card_pos, row_pos, 0] = np.clip((thr1 - data.feature_min[root_feat]) / data.feature_range[root_feat], 0.0, 1.0)
    child_mask = child_feat >= 0
    has_child[card_pos[child_mask], row_pos[child_mask]] = 1.0
    thr_target[card_pos[child_mask], row_pos[child_mask], 1] = np.clip(
        (thr2[child_mask] - data.feature_min[child_feat[child_mask]]) / data.feature_range[child_feat[child_mask]], 0.0, 1.0
    )
    weight_target[card_pos, row_pos, 0] = (w - w_mean) / w_std
    weight_target[card_pos, row_pos, 1] = (w_counter - w_mean) / w_std
    real_mask = 1.0 - pad_target
    thr1_mask = real_mask
    thr2_mask = real_mask * has_child
    weight_mask = real_mask
    tensor = np.concatenate(
        [pat_onehot, thr_target, weight_target, pad_target[..., None]], axis=-1
    ).astype(np.float32)
    row_dim = tensor.shape[-1]
    prior = cards_df.set_index("scorecard_id").loc[card_ids, "prior_log_odds"].to_numpy(np.float32)
    p_mean = float(prior.mean())
    p_std = float(prior.std()) or 1.0
    p_target = ((prior - p_mean) / p_std).astype(np.float32)
    return CorpusBundle(
        corpus_df=corpus_df, cards_df=cards_df, tree_rows_df=tree_rows, pattern_vocab=vocab, card_ids=card_ids,
        scorecard_tensor=tensor, pat_target=pat_target, thr_target=thr_target, weight_target=weight_target,
        pad_target=pad_target, has_child=has_child, real_mask=real_mask, thr1_mask=thr1_mask,
        thr2_mask=thr2_mask, weight_mask=weight_mask, prior_log_odds=prior, prior_target=p_target,
        n_sc=n_sc, t_max=t_max, n_rows=n_rows, n_pat=n_pat, row_dim=row_dim,
        weight_mean=w_mean, weight_std=w_std, prior_mean=p_mean, prior_std=p_std,
        prior_min=float(prior.min()), prior_max=float(prior.max()),
        best_initial_id=best_initial_id, best_initial_validation_accuracy=best_initial_validation_accuracy,
    )
