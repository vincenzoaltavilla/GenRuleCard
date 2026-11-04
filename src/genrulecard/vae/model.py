from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..core.artifacts import CorpusBundle
from ..core.config import VAEConfig
from .tree_matrix import (
    BOTH_CHILD, L_CHILD, N_TREE_TYPES, PAD_TREE, ROOT_W_L, ROOT_W_R,
    R_CHILD, STUMP, TreeMatrixTargets,
)


class ScoreCardDataset(Dataset):
    def __init__(self, corpus: CorpusBundle, indices: np.ndarray):
        self.corpus = corpus
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        card_index = int(self.indices[index])
        c = self.corpus
        return (
            torch.tensor(card_index, dtype=torch.long),
            torch.tensor(c.scorecard_tensor[card_index], dtype=torch.float32),
            torch.tensor(c.prior_target[card_index], dtype=torch.float32),
            torch.tensor(c.pat_target[card_index], dtype=torch.long),
            torch.tensor(c.thr_target[card_index], dtype=torch.float32),
            torch.tensor(c.weight_target[card_index], dtype=torch.float32),
            torch.tensor(c.pad_target[card_index], dtype=torch.float32),
            torch.tensor(c.thr1_mask[card_index], dtype=torch.float32),
            torch.tensor(c.thr2_mask[card_index], dtype=torch.float32),
            torch.tensor(c.weight_mask[card_index], dtype=torch.float32),
        )


class TreeScoreCardDataset(Dataset):
    def __init__(self, targets: TreeMatrixTargets, indices: np.ndarray):
        self.targets = targets
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        card_index = int(self.indices[index])
        t = self.targets
        return {
            "idx": torch.tensor(card_index, dtype=torch.long),
            "x": torch.tensor(t.tree_tensor[card_index], dtype=torch.float32),
            "tree_type": torch.tensor(t.tree_type[card_index], dtype=torch.long),
            "root_feat": torch.tensor(t.root_feat[card_index], dtype=torch.long),
            "left_child_feat": torch.tensor(t.left_child_feat[card_index], dtype=torch.long),
            "right_child_feat": torch.tensor(t.right_child_feat[card_index], dtype=torch.long),
            "thr": torch.tensor(t.thr[card_index], dtype=torch.float32),
            "weight": torch.tensor(t.weight[card_index], dtype=torch.float32),
            "root_mask": torch.tensor(t.root_mask[card_index], dtype=torch.float32),
            "left_child_mask": torch.tensor(t.left_child_mask[card_index], dtype=torch.float32),
            "right_child_mask": torch.tensor(t.right_child_mask[card_index], dtype=torch.float32),
            "thr_mask": torch.tensor(t.thr_mask[card_index], dtype=torch.float32),
            "weight_mask": torch.tensor(t.weight_mask[card_index], dtype=torch.float32),
            "prior": torch.tensor(t.prior[card_index], dtype=torch.float32),
            "target_f1": torch.tensor(t.target_f1[card_index], dtype=torch.float32),
            "target_leaves": torch.tensor(t.target_leaves[card_index], dtype=torch.float32),
            "target_signature": torch.tensor(t.target_signature[card_index], dtype=torch.float32),
        }


class ScoreCardVAE(nn.Module):
    def __init__(self, row_dim: int, n_rows: int, n_patterns: int, config: VAEConfig):
        super().__init__()
        self.n_rows = n_rows
        self.n_patterns = n_patterns
        hidden = config.hidden_dim
        input_dim = row_dim * n_rows + 1
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(config.dropout_p),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(config.dropout_p),
        )
        self.mu_head = nn.Linear(hidden // 2, config.latent_dim)
        self.logvar_head = nn.Linear(hidden // 2, config.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(config.dropout_p),
            nn.Linear(hidden // 2, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(config.dropout_p),
        )
        self.pattern_head = nn.Linear(hidden, n_rows * n_patterns)
        self.threshold_head = nn.Linear(hidden, n_rows * 2)
        self.weight_head = nn.Linear(hidden, n_rows * 2)
        self.padding_head = nn.Linear(hidden, n_rows)
        self.prior_head = nn.Linear(hidden, 1)

    def encode(self, matrix: torch.Tensor, prior: torch.Tensor):
        flat = matrix.flatten(start_dim=1)
        hidden = self.encoder(torch.cat([flat, prior.unsqueeze(1)], dim=1))
        return self.mu_head(hidden), self.logvar_head(hidden).clamp(-8.0, 8.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.decoder(z)
        batch = z.shape[0]
        return {
            "pattern_logits": self.pattern_head(hidden).reshape(batch, self.n_rows, self.n_patterns),
            "thresholds": torch.sigmoid(self.threshold_head(hidden).reshape(batch, self.n_rows, 2)),
            "weights": self.weight_head(hidden).reshape(batch, self.n_rows, 2),
            "padding_logits": self.padding_head(hidden),
            "prior": self.prior_head(hidden).squeeze(-1),
        }

    def forward(self, matrix: torch.Tensor, prior: torch.Tensor, sample: bool = True):
        mu, logvar = self.encode(matrix, prior)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z), mu, logvar


class TreeMatrixVAE(nn.Module):
    def __init__(
        self,
        tree_input_dim: int,
        n_trees: int,
        n_features: int,
        config: VAEConfig,
    ):
        super().__init__()
        self.n_trees = int(n_trees)
        self.n_features = int(n_features)
        hidden = int(config.hidden_dim)
        self.input_proj = nn.Linear(tree_input_dim, hidden)
        self.pos = nn.Parameter(torch.randn(self.n_trees, hidden) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.n_heads,
            dim_feedforward=hidden * 3,
            dropout=config.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.prior_proj = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.LayerNorm(hidden),
        )
        self.mu_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, config.latent_dim))
        self.logvar_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, config.latent_dim))

        self.z_proj = nn.Linear(config.latent_dim, hidden)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.n_heads,
            dim_feedforward=hidden * 3,
            dropout=config.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.n_layers)
        self.type_head = nn.Linear(hidden, N_TREE_TYPES)
        self.root_head = nn.Linear(hidden, self.n_features)
        self.left_child_head = nn.Linear(hidden, self.n_features)
        self.right_child_head = nn.Linear(hidden, self.n_features)
        self.threshold_head = nn.Linear(hidden, 3)
        self.weight_head = nn.Linear(hidden, 6)
        self.prior_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def encode(self, matrix: torch.Tensor, prior: torch.Tensor):
        hidden = self.input_proj(matrix) + self.pos[None, :, :]
        hidden = self.encoder(hidden)
        pooled = hidden.mean(dim=1) + self.prior_proj(prior.unsqueeze(1))
        return self.mu_head(pooled), self.logvar_head(pooled).clamp(-8.0, 8.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.z_proj(z)[:, None, :] + self.pos[None, :, :]
        hidden = self.decoder(hidden)
        pooled = hidden.mean(dim=1)
        return {
            "type_logits": self.type_head(hidden),
            "root_logits": self.root_head(hidden),
            "left_child_logits": self.left_child_head(hidden),
            "right_child_logits": self.right_child_head(hidden),
            "thr": torch.sigmoid(self.threshold_head(hidden)),
            "weight": self.weight_head(hidden),
            "prior": self.prior_head(pooled).squeeze(-1),
        }

    def forward(self, matrix: torch.Tensor, prior: torch.Tensor, sample: bool = True):
        mu, logvar = self.encode(matrix, prior)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z), mu, logvar


class LeafMetricTreeVAE(nn.Module):
    """MLP z-only usato dal protocollo VAE_fmi_f1_sig.

    Il prior entra soltanto nell'encoder; il decoder riceve esclusivamente z
    e deve quindi ricostruire autonomamente anche il prior.
    """

    def __init__(self, tree_input_dim: int, n_trees: int, n_features: int, config: VAEConfig):
        super().__init__()
        self.n_trees = int(n_trees)
        self.n_features = int(n_features)
        hidden_dims = tuple(config.hidden_dims or (config.hidden_dim,) * config.n_layers)
        flat_dim = self.n_trees * int(tree_input_dim) + 1
        encoder = []
        previous = flat_dim
        for hidden in hidden_dims:
            encoder.extend([
                nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                nn.Dropout(config.dropout_p),
            ])
            previous = hidden
        self.encoder = nn.Sequential(*encoder)
        self.mu_head = nn.Linear(previous, config.latent_dim)
        self.logvar_head = nn.Linear(previous, config.latent_dim)

        decoder = []
        previous = config.latent_dim
        for hidden in reversed(hidden_dims):
            decoder.extend([
                nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                nn.Dropout(config.dropout_p),
            ])
            previous = hidden
        self.decoder = nn.Sequential(*decoder)
        self.type_head = nn.Linear(previous, self.n_trees * N_TREE_TYPES)
        self.root_head = nn.Linear(previous, self.n_trees * self.n_features)
        self.left_child_head = nn.Linear(previous, self.n_trees * self.n_features)
        self.right_child_head = nn.Linear(previous, self.n_trees * self.n_features)
        self.threshold_head = nn.Linear(previous, self.n_trees * 3)
        self.weight_head = nn.Linear(previous, self.n_trees * 6)
        self.prior_head = nn.Linear(previous, 1)

    def encode(self, matrix: torch.Tensor, prior: torch.Tensor):
        batch = matrix.shape[0]
        hidden = self.encoder(torch.cat([matrix.reshape(batch, -1), prior.reshape(batch, 1)], dim=1))
        return self.mu_head(hidden), self.logvar_head(hidden).clamp(-8.0, 6.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.decoder(z)
        batch = z.shape[0]
        return {
            "type_logits": self.type_head(hidden).reshape(batch, self.n_trees, N_TREE_TYPES),
            "root_logits": self.root_head(hidden).reshape(batch, self.n_trees, self.n_features),
            "left_child_logits": self.left_child_head(hidden).reshape(batch, self.n_trees, self.n_features),
            "right_child_logits": self.right_child_head(hidden).reshape(batch, self.n_trees, self.n_features),
            "thr": torch.sigmoid(self.threshold_head(hidden)).reshape(batch, self.n_trees, 3),
            "weight": self.weight_head(hidden).reshape(batch, self.n_trees, 6),
            "prior": self.prior_head(hidden).reshape(batch),
        }

    def forward(self, matrix: torch.Tensor, prior: torch.Tensor, sample: bool = True):
        mu, logvar = self.encode(matrix, prior)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z), mu, logvar


def build_vae(
    corpus: CorpusBundle,
    config: VAEConfig,
    device: torch.device,
    data=None,
) -> nn.Module:
    if getattr(config, "representation", "row") == "tree":
        if data is None:
            raise ValueError("vae.representation='tree' richiede il dataset processato per conoscere le feature.")
        from .tree_matrix import tree_input_dim

        model_class = LeafMetricTreeVAE if config.vae_profile == "fmi_f1_sig" else TreeMatrixVAE
        return model_class(
            tree_input_dim(len(data.feature_names)),
            corpus.t_max,
            len(data.feature_names),
            config,
        ).to(device)
    return ScoreCardVAE(corpus.row_dim, corpus.n_rows, corpus.n_pat, config).to(device)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < prediction.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(prediction)
    return ((prediction - target).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)


def beta_schedule(epoch: int, config: VAEConfig) -> float:
    warmup = max(1, int(config.epochs * config.beta_warmup_fraction))
    return config.beta_max * min(1.0, epoch / warmup)


def _zscore(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)


def pairwise_metric_loss(mu: torch.Tensor, card_indices: torch.Tensor, target_distance: torch.Tensor) -> torch.Tensor:
    if len(card_indices) < 3:
        return torch.zeros((), device=mu.device)
    target_distance = target_distance[card_indices][:, card_indices]
    latent_distance = torch.cdist(mu, mu, p=2)
    # Metric loss is deterministic: it uses mu, not sampled z, and excludes i == j.
    upper = torch.triu(torch.ones_like(target_distance, dtype=torch.bool), diagonal=1)
    target, latent = target_distance[upper], latent_distance[upper]
    return F.mse_loss(_zscore(latent), _zscore(target))


def fmi_f1_sig_metric_loss(
    mu: torch.Tensor,
    card_indices: torch.Tensor,
    target_distance: torch.Tensor,
    config: VAEConfig,
) -> torch.Tensor:
    if len(card_indices) < 3:
        return torch.zeros((), device=mu.device)
    target_matrix = target_distance[card_indices][:, card_indices]
    latent_matrix = torch.cdist(mu, mu)
    latent_matrix = latent_matrix / latent_matrix.detach().mean().clamp_min(1e-6)
    target_matrix = target_matrix / target_matrix.mean().clamp_min(1e-6)
    upper = torch.triu(torch.ones_like(target_matrix, dtype=torch.bool), diagonal=1)
    latent, target = latent_matrix[upper], target_matrix[upper]
    distance_loss = F.smooth_l1_loss(latent, target)
    latent_centered, target_centered = latent - latent.mean(), target - target.mean()
    pearson = torch.sum(latent_centered * target_centered) / (
        torch.linalg.vector_norm(latent_centered) * torch.linalg.vector_norm(target_centered)
    ).clamp_min(1e-8)
    n_rank = min(8192, max(1, latent.numel() * 4))
    rank_ids = torch.arange(n_rank, device=mu.device)
    first = (37 * rank_ids + 17) % latent.numel()
    second = (101 * rank_ids + 29) % latent.numel()
    target_gap = target[first] - target[second]
    valid = target_gap.abs() >= config.metric_rank_min_gap
    rank_loss = torch.zeros((), device=mu.device)
    if valid.any():
        latent_gap = latent[first[valid]] - latent[second[valid]]
        rank_loss = F.softplus(
            -target_gap[valid].sign() * latent_gap / config.metric_rank_temperature
        ).mean()
    return (
        distance_loss
        + config.metric_pearson_weight * (1.0 - pearson)
        + config.metric_rank_weight * rank_loss
    )


def soft_tree_f1(
    output: dict[str, torch.Tensor], X_validation: torch.Tensor, y_validation: torch.Tensor,
    temperature: float = 0.08,
) -> torch.Tensor:
    feature_probability = F.softmax(output["root_logits"], dim=-1)
    threshold = output["thr"][..., 0]
    active = 1.0 - F.softmax(output["type_logits"], dim=-1)[..., PAD_TREE]
    x_feature = torch.einsum("btk,nk->btn", feature_probability, X_validation)
    go_right = torch.sigmoid((x_feature - threshold.unsqueeze(-1)) / temperature)
    contribution = (
        (1.0 - go_right) * output["weight"][..., ROOT_W_L].unsqueeze(-1)
        + go_right * output["weight"][..., ROOT_W_R].unsqueeze(-1)
    )
    logits = output["prior"].unsqueeze(-1) + (active.unsqueeze(-1) * contribution).sum(dim=1)
    probability = torch.sigmoid(logits)
    target = y_validation.unsqueeze(0).expand_as(probability)
    tp = (probability * target).sum(dim=1)
    fp = (probability * (1.0 - target)).sum(dim=1)
    fn = ((1.0 - probability) * target).sum(dim=1)
    return (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)


def local_pairwise_metric_loss(
    mu: torch.Tensor,
    card_indices: torch.Tensor,
    target_distance: torch.Tensor,
) -> torch.Tensor:
    if len(card_indices) < 3:
        return torch.zeros((), device=mu.device)
    target_distance = target_distance[card_indices][:, card_indices]
    latent_distance = torch.cdist(mu, mu, p=2)
    upper = torch.triu(torch.ones_like(target_distance, dtype=torch.bool), diagonal=1)
    target, latent = target_distance[upper], latent_distance[upper]
    if target.numel() < 2:
        return torch.zeros((), device=mu.device)
    # Local metric learning: pairs that are close according to the target
    # distance matter more, because the experiment measures preservation of
    # nearest FMI neighbours, not only global rank correlation.
    target_z = _zscore(target)
    latent_z = _zscore(latent)
    weights = torch.softmax(-target_z, dim=0) * target.numel()
    return ((latent_z - target_z).pow(2) * weights).mean()


def tree_local_pairwise_metric_loss(
    mu: torch.Tensor,
    card_indices: torch.Tensor,
    target_distance: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if len(card_indices) < 3:
        return torch.zeros((), device=mu.device)
    target_distance = target_distance[card_indices][:, card_indices]
    latent_distance = torch.cdist(mu, mu, p=2)
    upper = torch.triu(torch.ones_like(target_distance, dtype=torch.bool), diagonal=1)
    target, latent = target_distance[upper], latent_distance[upper]
    if target.numel() < 2:
        return torch.zeros((), device=mu.device)
    weights = torch.softmax(-target / float(temperature), dim=0) * target.numel()
    return ((_zscore(latent) - _zscore(target)).pow(2) * weights).mean()


def tree_masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < prediction.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(prediction)
    values = F.smooth_l1_loss(prediction, target, reduction="none") * mask
    return values.sum() / mask.sum().clamp_min(1.0)


def tree_total_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    target_distance: torch.Tensor,
    config: VAEConfig,
    X_validation: torch.Tensor | None = None,
    y_validation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = output["type_logits"].device
    batch_size, n_trees, _ = output["type_logits"].shape
    indices = batch["idx"].to(device)
    tree_type = batch["tree_type"].to(device)
    l_type = F.cross_entropy(
        output["type_logits"].reshape(batch_size * n_trees, N_TREE_TYPES),
        tree_type.reshape(-1),
    )

    def ce_masked(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(logits.device) > 0.5
        if not mask.any():
            return torch.zeros((), device=logits.device)
        return F.cross_entropy(logits[mask], target.to(logits.device)[mask])

    l_root = ce_masked(output["root_logits"], batch["root_feat"], batch["root_mask"])
    l_left_child = ce_masked(
        output["left_child_logits"], batch["left_child_feat"], batch["left_child_mask"]
    )
    l_right_child = ce_masked(
        output["right_child_logits"], batch["right_child_feat"], batch["right_child_mask"]
    )
    l_threshold = tree_masked_smooth_l1(
        output["thr"], batch["thr"].to(device), batch["thr_mask"].to(device)
    )
    l_weight = tree_masked_smooth_l1(
        output["weight"], batch["weight"].to(device), batch["weight_mask"].to(device)
    )
    l_prior = F.smooth_l1_loss(output["prior"], batch["prior"].to(device))
    l_kl = (
        -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        if config.vae_profile == "fmi_f1_sig"
        else -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    )
    l_metric = (
        fmi_f1_sig_metric_loss(mu, indices, target_distance, config)
        if config.vae_profile == "fmi_f1_sig"
        else pairwise_metric_loss(mu, indices, target_distance)
    )
    l_local_metric = tree_local_pairwise_metric_loss(
        mu, indices, target_distance, config.tree_local_metric_temperature
    )

    lw = config.loss_weights
    l_count_leaves = torch.zeros((), device=device)
    l_signature = torch.zeros((), device=device)
    l_f1 = torch.zeros((), device=device)
    if config.vae_profile == "fmi_f1_sig":
        temperature = config.signature_temperature
        type_probability = F.softmax(output["type_logits"] / temperature, dim=-1)
        leaves_per_type = output["type_logits"].new_tensor([0.0, 2.0, 3.0, 3.0, 4.0])
        expected_leaves = torch.einsum("btk,k->b", type_probability, leaves_per_type)
        target_leaves = batch["target_leaves"].to(device)
        l_count_leaves = (
            (expected_leaves - target_leaves).abs() / target_leaves.clamp_min(1.0)
        ).mean()

        root_probability = F.softmax(output["root_logits"] / temperature, dim=-1)
        left_probability = F.softmax(output["left_child_logits"] / temperature, dim=-1)
        right_probability = F.softmax(output["right_child_logits"] / temperature, dim=-1)
        p_stump, p_left = type_probability[..., STUMP], type_probability[..., L_CHILD]
        p_right, p_both = type_probability[..., R_CHILD], type_probability[..., BOTH_CHILD]
        simple_le = torch.einsum("bt,btf->bf", p_stump + p_right, root_probability)
        simple_gt = torch.einsum("bt,btf->bf", p_stump + p_left, root_probability)
        joint_left = torch.einsum("bt,btr,btc->brc", p_left + p_both, root_probability, left_probability)
        joint_right = torch.einsum("bt,btr,btc->brc", p_right + p_both, root_probability, right_probability)
        predicted_signature = torch.cat([
            simple_le, simple_gt, joint_left.flatten(1), joint_left.flatten(1),
            joint_right.flatten(1), joint_right.flatten(1),
        ], dim=1) / max(1.0, float(n_trees))
        target_signature = batch["target_signature"].to(device)
        l_signature = (
            (predicted_signature - target_signature).abs().sum(dim=1)
            / target_signature.sum(dim=1).clamp_min(1.0 / max(1.0, float(n_trees)))
        ).mean()
        if X_validation is None or y_validation is None:
            raise ValueError("Il profilo fmi_f1_sig richiede validation X/y per la loss F1.")
        soft_f1 = soft_tree_f1(output, X_validation, y_validation)
        l_f1 = F.smooth_l1_loss(soft_f1, batch["target_f1"].to(device))
    reconstruction = (
        lw.get("type", 4.0) * l_type
        + lw.get("feature", lw.get("root", 3.0)) * (l_root + 0.5 * (l_left_child + l_right_child))
        + lw.get("threshold", 3.0) * l_threshold
        + lw.get("weight", 3.0) * l_weight
        + lw.get("prior", 1.0) * l_prior
    )
    total = (
        reconstruction
        + beta * l_kl
        + lw.get("count_leaves", 0.0) * l_count_leaves
        + lw.get("signature_count", 0.0) * l_signature
        + lw.get("f1", 0.0) * l_f1
        + lw.get("metric", 0.50) * l_metric
        + lw.get("local_metric", 0.25) * l_local_metric
    )
    return total, {
        "total": total,
        "reconstruction": reconstruction,
        "type": l_type,
        "root": l_root,
        "left_child": l_left_child,
        "right_child": l_right_child,
        "threshold": l_threshold,
        "weight": l_weight,
        "prior": l_prior,
        "kl": l_kl,
        "metric": l_metric,
        "local_metric": l_local_metric,
        "count_leaves": l_count_leaves,
        "signature_count": l_signature,
        "f1_loss": l_f1,
    }


def total_loss(
    output: dict[str, torch.Tensor], mu: torch.Tensor, logvar: torch.Tensor, card_indices: torch.Tensor,
    pattern: torch.Tensor, threshold: torch.Tensor, weight: torch.Tensor, padding: torch.Tensor,
    mask_thr1: torch.Tensor, mask_thr2: torch.Tensor, mask_weight: torch.Tensor,
    prior: torch.Tensor, beta: float, pattern_class_weight: torch.Tensor,
    target_distance: torch.Tensor, config: VAEConfig,
    loss_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch, rows, patterns = output["pattern_logits"].shape
    valid = pattern.reshape(-1) >= 0
    l_pattern = F.cross_entropy(
        output["pattern_logits"].reshape(batch * rows, patterns)[valid],
        pattern.reshape(-1)[valid], weight=pattern_class_weight,
    )
    l_thr1 = masked_mse(output["thresholds"][..., 0], threshold[..., 0], mask_thr1)
    l_thr2 = masked_mse(output["thresholds"][..., 1], threshold[..., 1], mask_thr2)
    l_padding = F.binary_cross_entropy_with_logits(output["padding_logits"], padding)
    pred_real_count = torch.sigmoid(-output["padding_logits"]).sum(dim=1)
    target_real_count = (1.0 - padding).sum(dim=1)
    l_count = F.mse_loss(pred_real_count / rows, target_real_count / rows)
    l_weight = masked_mse(output["weights"], weight, mask_weight)
    l_prior = F.mse_loss(output["prior"], prior)
    l_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    l_metric = pairwise_metric_loss(mu, card_indices, target_distance)
    l_local_metric = local_pairwise_metric_loss(mu, card_indices, target_distance)
    lw = config.loss_weights if loss_weights is None else loss_weights
    count_weight = lw.get("count", 0.0)
    local_metric_weight = lw.get("local_metric", 0.0)
    total = (
        lw["pattern"] * l_pattern + lw["padding"] * l_padding + lw["thr1"] * l_thr1 +
        lw["thr2"] * l_thr2 + lw["weight"] * l_weight + lw["prior"] * l_prior +
        count_weight * l_count + beta * l_kl + lw["metric"] * l_metric +
        local_metric_weight * l_local_metric
    )
    reconstruction = (
        lw["pattern"] * l_pattern + lw["padding"] * l_padding
        + lw["thr1"] * l_thr1 + lw["thr2"] * l_thr2
        + lw["weight"] * l_weight + lw["prior"] * l_prior
        + count_weight * l_count
    )
    return total, {
        "total": total, "pattern": l_pattern, "padding": l_padding, "thr1": l_thr1,
        "thr2": l_thr2, "weight": l_weight, "prior": l_prior, "kl": l_kl,
        "metric": l_metric, "local_metric": l_local_metric, "count": l_count,
        "reconstruction": reconstruction,
    }
