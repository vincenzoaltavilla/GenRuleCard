from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.artifacts import CorpusBundle, ProcessedDataset
from ..rulecard_fit.encoding import make_real_row, padding_row, rows_to_leaves


TREE_TYPES = ["PAD", "STUMP", "L_CHILD", "R_CHILD", "BOTH_CHILD"]
PAD_TREE, STUMP, L_CHILD, R_CHILD, BOTH_CHILD = range(len(TREE_TYPES))
N_TREE_TYPES = len(TREE_TYPES)

ROOT_W_L, ROOT_W_R, LCH_W_L, LCH_W_R, RCH_W_L, RCH_W_R = range(6)
TREE_CONT_DIM = 3 + 6


@dataclass
class TreeMatrixTargets:
    tree_tensor: np.ndarray
    tree_type: np.ndarray
    root_feat: np.ndarray
    left_child_feat: np.ndarray
    right_child_feat: np.ndarray
    thr: np.ndarray
    weight: np.ndarray
    root_mask: np.ndarray
    left_child_mask: np.ndarray
    right_child_mask: np.ndarray
    thr_mask: np.ndarray
    weight_mask: np.ndarray
    prior: np.ndarray
    target_f1: np.ndarray
    target_leaves: np.ndarray
    target_signature: np.ndarray
    n_features: int
    n_trees: int
    tree_input_dim: int


def tree_input_dim(n_features: int) -> int:
    return N_TREE_TYPES + 3 * int(n_features) + TREE_CONT_DIM


def canonicalize_corpus_tree_order(corpus: CorpusBundle) -> CorpusBundle:
    """Rende confrontabili gli slot posizionali senza cambiare l'ensemble."""
    corpus.corpus_df = corpus.corpus_df.copy()
    corpus.tree_rows_df = corpus.tree_rows_df.copy()
    position_map: dict[tuple[int, int], int] = {}
    for scorecard_id, card_leaves in corpus.corpus_df.groupby("scorecard_id", sort=False):
        records = []
        for old_tree_idx, tree_leaves in card_leaves.groupby("tree_idx", sort=False):
            structural = tuple(sorted(
                tuple((int(condition[0]), "L" if str(condition[2]) in {"<", "<="} else "R")
                      for condition in path)
                for path in tree_leaves["conditions_key"]
            ))
            thresholds = tuple(sorted(
                tuple(round(float(condition[1]), 8) for condition in path)
                for path in tree_leaves["conditions_key"]
            ))
            weights = tuple(sorted(round(float(value), 8) for value in tree_leaves["w"]))
            records.append((structural, thresholds, weights, int(old_tree_idx)))
        ordered = [record[-1] for record in sorted(records)]
        remaining = sorted(set(range(int(corpus.t_max))) - set(ordered))
        for new_tree_idx, old_tree_idx in enumerate(ordered + remaining):
            position_map[(int(scorecard_id), int(old_tree_idx))] = int(new_tree_idx)
    for frame in (corpus.corpus_df, corpus.tree_rows_df):
        frame["tree_idx"] = [
            position_map[(int(card_id), int(tree_idx))]
            for card_id, tree_idx in zip(frame["scorecard_id"], frame["tree_idx"])
        ]
    corpus.corpus_df = corpus.corpus_df.sort_values(
        ["scorecard_id", "tree_idx", "local_rule_idx"]
    ).reset_index(drop=True)
    corpus.tree_rows_df = corpus.tree_rows_df.sort_values(
        ["scorecard_id", "tree_idx", "row_in_tree"]
    ).reset_index(drop=True)
    return corpus


def build_tree_matrix_targets(corpus: CorpusBundle, data: ProcessedDataset) -> TreeMatrixTargets:
    """Tree-level representation: one row per boosting tree.

    The row contains:
    - tree type one-hot: PAD, STUMP, left-child, right-child, both-child;
    - root feature one-hot;
    - optional left-child feature one-hot;
    - optional right-child feature one-hot;
    - three normalized thresholds: root, left child, right child;
    - six normalized leaf weights.
    """
    n_features = len(data.feature_names)
    n_trees = int(corpus.t_max)
    n = int(corpus.n_sc)

    def norm_thr(value: float, feature: int) -> float:
        return float(
            np.clip(
                (float(value) - data.feature_min[int(feature)]) / data.feature_range[int(feature)],
                0.0,
                1.0,
            )
        )

    def norm_weight(value: float) -> float:
        return float((float(value) - corpus.weight_mean) / corpus.weight_std)

    tree_type = np.zeros((n, n_trees), dtype=np.int64)
    root_feat = np.zeros((n, n_trees), dtype=np.int64)
    left_child_feat = np.zeros((n, n_trees), dtype=np.int64)
    right_child_feat = np.zeros((n, n_trees), dtype=np.int64)
    thr = np.zeros((n, n_trees, 3), dtype=np.float32)
    weight = np.zeros((n, n_trees, 6), dtype=np.float32)

    root_mask = np.zeros((n, n_trees), dtype=np.float32)
    left_child_mask = np.zeros((n, n_trees), dtype=np.float32)
    right_child_mask = np.zeros((n, n_trees), dtype=np.float32)
    thr_mask = np.zeros((n, n_trees, 3), dtype=np.float32)
    weight_mask = np.zeros((n, n_trees, 6), dtype=np.float32)

    card_to_pos = {identifier: position for position, identifier in enumerate(corpus.card_ids)}
    rows = corpus.tree_rows_df.copy()
    for scorecard_id in corpus.card_ids:
        card_pos = card_to_pos[int(scorecard_id)]
        card_rows = rows.loc[rows["scorecard_id"] == scorecard_id]
        for tree_idx in range(n_trees):
            tree = card_rows.loc[card_rows["tree_idx"] == tree_idx].sort_values("row_in_tree")
            real = tree.loc[tree["is_padding"] == 0].sort_values("row_in_tree")
            if real.empty:
                tree_type[card_pos, tree_idx] = PAD_TREE
                continue

            first = real.iloc[0]
            root = int(first["root_feat"])
            root_feat[card_pos, tree_idx] = root
            thr[card_pos, tree_idx, 0] = norm_thr(float(first["thr1"]), root)
            root_mask[card_pos, tree_idx] = 1.0
            thr_mask[card_pos, tree_idx, 0] = 1.0

            by_row = {int(row.row_in_tree): row for row in real.itertuples(index=False)}
            left = by_row.get(0)
            right = by_row.get(1)

            if len(real) == 1:
                tree_type[card_pos, tree_idx] = STUMP
                weight[card_pos, tree_idx, ROOT_W_L] = norm_weight(first["w"])
                weight[card_pos, tree_idx, ROOT_W_R] = norm_weight(first["w_counter"])
                weight_mask[card_pos, tree_idx, [ROOT_W_L, ROOT_W_R]] = 1.0
                continue

            left_has_child = left is not None and int(left.child_feat) >= 0
            right_has_child = right is not None and int(right.child_feat) >= 0
            if left_has_child and right_has_child:
                tree_type[card_pos, tree_idx] = BOTH_CHILD
            elif left_has_child:
                tree_type[card_pos, tree_idx] = L_CHILD
            elif right_has_child:
                tree_type[card_pos, tree_idx] = R_CHILD
            else:
                tree_type[card_pos, tree_idx] = STUMP

            if left is not None:
                if left_has_child:
                    child = int(left.child_feat)
                    left_child_feat[card_pos, tree_idx] = child
                    thr[card_pos, tree_idx, 1] = norm_thr(float(left.thr2), child)
                    left_child_mask[card_pos, tree_idx] = 1.0
                    thr_mask[card_pos, tree_idx, 1] = 1.0
                    weight[card_pos, tree_idx, LCH_W_L] = norm_weight(left.w)
                    weight[card_pos, tree_idx, LCH_W_R] = norm_weight(left.w_counter)
                    weight_mask[card_pos, tree_idx, [LCH_W_L, LCH_W_R]] = 1.0
                else:
                    weight[card_pos, tree_idx, ROOT_W_L] = norm_weight(left.w)
                    weight_mask[card_pos, tree_idx, ROOT_W_L] = 1.0

            if right is not None:
                if right_has_child:
                    child = int(right.child_feat)
                    right_child_feat[card_pos, tree_idx] = child
                    thr[card_pos, tree_idx, 2] = norm_thr(float(right.thr2), child)
                    right_child_mask[card_pos, tree_idx] = 1.0
                    thr_mask[card_pos, tree_idx, 2] = 1.0
                    weight[card_pos, tree_idx, RCH_W_L] = norm_weight(right.w)
                    weight[card_pos, tree_idx, RCH_W_R] = norm_weight(right.w_counter)
                    weight_mask[card_pos, tree_idx, [RCH_W_L, RCH_W_R]] = 1.0
                else:
                    weight[card_pos, tree_idx, ROOT_W_R] = norm_weight(right.w)
                    weight_mask[card_pos, tree_idx, ROOT_W_R] = 1.0

            if tree_type[card_pos, tree_idx] == STUMP and right is not None and not left_has_child and not right_has_child:
                weight_mask[card_pos, tree_idx, [ROOT_W_L, ROOT_W_R]] = 1.0

    type_oh = np.eye(N_TREE_TYPES, dtype=np.float32)[tree_type]
    root_oh = np.eye(n_features, dtype=np.float32)[root_feat]
    left_child_oh = np.eye(n_features, dtype=np.float32)[left_child_feat]
    right_child_oh = np.eye(n_features, dtype=np.float32)[right_child_feat]
    tree_tensor = np.concatenate(
        [type_oh, root_oh, left_child_oh, right_child_oh, thr, weight],
        axis=-1,
    ).astype(np.float32)

    cards = corpus.cards_df.set_index("scorecard_id").reindex(corpus.card_ids)
    target_f1 = cards["validation_f1"].to_numpy(dtype=np.float32)
    target_leaves = (
        corpus.corpus_df.groupby("scorecard_id").size()
        .reindex(corpus.card_ids).to_numpy(dtype=np.float32)
    )
    signature_dim = 2 * n_features + 4 * n_features * n_features
    target_signature = np.zeros((n, signature_dim), dtype=np.float32)
    card_to_position = {int(identifier): position for position, identifier in enumerate(corpus.card_ids)}
    for row in corpus.corpus_df.itertuples(index=False):
        path = row.conditions_key
        if not path:
            continue
        root_feature, _, root_operator = path[0][:3]
        root_feature = int(root_feature)
        root_side = 0 if str(root_operator) in {"<", "<="} else 1
        if not 0 <= root_feature < n_features:
            continue
        if len(path) == 1:
            signature_index = root_side * n_features + root_feature
        else:
            child_feature, _, child_operator = path[1][:3]
            child_feature = int(child_feature)
            if not 0 <= child_feature < n_features:
                continue
            child_side = 0 if str(child_operator) in {"<", "<="} else 1
            pair_kind = 2 * root_side + child_side
            signature_index = (
                2 * n_features + pair_kind * n_features * n_features
                + root_feature * n_features + child_feature
            )
        target_signature[card_to_position[int(row.scorecard_id)], signature_index] += 1.0
    target_signature /= max(1.0, float(n_trees))

    return TreeMatrixTargets(
        tree_tensor=tree_tensor,
        tree_type=tree_type,
        root_feat=root_feat,
        left_child_feat=left_child_feat,
        right_child_feat=right_child_feat,
        thr=thr,
        weight=weight,
        root_mask=root_mask,
        left_child_mask=left_child_mask,
        right_child_mask=right_child_mask,
        thr_mask=thr_mask,
        weight_mask=weight_mask,
        prior=corpus.prior_target.astype(np.float32),
        target_f1=target_f1,
        target_leaves=target_leaves,
        target_signature=target_signature,
        n_features=n_features,
        n_trees=n_trees,
        tree_input_dim=tree_input_dim(n_features),
    )


def decode_tree_output_to_rows(
    output: dict[str, np.ndarray],
    prior_norm: float,
    scorecard_id: int,
    corpus: CorpusBundle,
    data: ProcessedDataset,
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict]:
    """Natural TreeVAE decoding: argmax structure/features + continuous thresholds/weights."""
    type_logits = output["type_logits"]
    root_logits = output["root_logits"]
    left_child_logits = output["left_child_logits"]
    right_child_logits = output["right_child_logits"]
    thr = output["thr"]
    weight = output["weight"]
    n_trees = int(type_logits.shape[0])

    def denorm_threshold(value: float, feature: int) -> float:
        return float(
            data.feature_min[int(feature)]
            + float(np.clip(value, 0.0, 1.0)) * data.feature_range[int(feature)]
        )

    def denorm_weight(value: float) -> float:
        return float(value * corpus.weight_std + corpus.weight_mean)

    rows: list[dict] = []
    n_active = 0
    for tree_idx in range(n_trees):
        decoded_type = int(np.argmax(type_logits[tree_idx]))
        if decoded_type == PAD_TREE:
            rows.extend([
                padding_row(scorecard_id, tree_idx, 0),
                padding_row(scorecard_id, tree_idx, 1),
            ])
            continue
        n_active += 1
        root = int(np.argmax(root_logits[tree_idx]))
        root_thr = denorm_threshold(thr[tree_idx, 0], root)
        if decoded_type == STUMP:
            rows.append(make_real_row(
                scorecard_id, tree_idx, 0, "STUMP", root, -1, root_thr, 0.0,
                denorm_weight(weight[tree_idx, ROOT_W_L]),
                denorm_weight(weight[tree_idx, ROOT_W_R]),
                data.feature_names,
            ))
            rows.append(padding_row(scorecard_id, tree_idx, 1))
            continue
        if decoded_type == L_CHILD:
            child = int(np.argmax(left_child_logits[tree_idx]))
            child_thr = denorm_threshold(thr[tree_idx, 1], child)
            rows.append(make_real_row(
                scorecard_id, tree_idx, 0, "L", root, child, root_thr, child_thr,
                denorm_weight(weight[tree_idx, LCH_W_L]),
                denorm_weight(weight[tree_idx, LCH_W_R]),
                data.feature_names,
            ))
            rows.append(make_real_row(
                scorecard_id, tree_idx, 1, "R_LEAF", root, -1, root_thr, 0.0,
                denorm_weight(weight[tree_idx, ROOT_W_R]),
                denorm_weight(weight[tree_idx, ROOT_W_R]),
                data.feature_names,
            ))
            continue
        if decoded_type == R_CHILD:
            child = int(np.argmax(right_child_logits[tree_idx]))
            child_thr = denorm_threshold(thr[tree_idx, 2], child)
            rows.append(make_real_row(
                scorecard_id, tree_idx, 0, "L_LEAF", root, -1, root_thr, 0.0,
                denorm_weight(weight[tree_idx, ROOT_W_L]),
                denorm_weight(weight[tree_idx, ROOT_W_L]),
                data.feature_names,
            ))
            rows.append(make_real_row(
                scorecard_id, tree_idx, 1, "R", root, child, root_thr, child_thr,
                denorm_weight(weight[tree_idx, RCH_W_L]),
                denorm_weight(weight[tree_idx, RCH_W_R]),
                data.feature_names,
            ))
            continue
        left_child = int(np.argmax(left_child_logits[tree_idx]))
        right_child = int(np.argmax(right_child_logits[tree_idx]))
        left_thr = denorm_threshold(thr[tree_idx, 1], left_child)
        right_thr = denorm_threshold(thr[tree_idx, 2], right_child)
        rows.append(make_real_row(
            scorecard_id, tree_idx, 0, "L", root, left_child, root_thr, left_thr,
            denorm_weight(weight[tree_idx, LCH_W_L]),
            denorm_weight(weight[tree_idx, LCH_W_R]),
            data.feature_names,
        ))
        rows.append(make_real_row(
            scorecard_id, tree_idx, 1, "R", root, right_child, root_thr, right_thr,
            denorm_weight(weight[tree_idx, RCH_W_L]),
            denorm_weight(weight[tree_idx, RCH_W_R]),
            data.feature_names,
        ))

    rows_df = pd.DataFrame(rows)
    leaves = rows_to_leaves(rows_df, data.feature_names)
    prior = float(prior_norm * corpus.prior_std + corpus.prior_mean)
    prior = float(np.clip(prior, corpus.prior_min, corpus.prior_max))
    return rows_df, leaves, prior, {
        "n_active": int(n_active),
        "n_rules_prefix": int(len(leaves)),
        "target_n_rules": None,
        "predicted_active": float(n_active),
        "decode_count_penalty": 0.0,
        "tree_decoder": "natural_argmax",
    }
