from tqdm import tqdm
from models.f_losses import moco_contrastive_loss, supervised_moco_loss, get_moco_mrr, triplet_loss, retrieval_loss, retrieval_metrics_val, best_diagonal_window
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score
from gradient import gradient_cosine, get_loss_gradients, get_gradient_stats, gradient_norm, safe_mean


def train_vit_moco_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        anchor_work_id = batch["src_work_id"].to(device)
        pos_work_id = batch["pos_work_id"].to(device)
        # neg_work_id = batch["neg_work_id"].to(device)

        anchor_perf_id = batch["src_label"].to(device)
        pos_perf_id = batch["pos_label"].to(device)
        # neg_perf_id = batch["neg_label"].to(device)

        anchor = batch["orig_inp"].to(device)
        positive = batch["pos_inp"].to(device)
        # negative = batch["neg_inp"].to(device)

        output = model(anchor, positive)

        anchor_feature = output["x1_glob_feature"]
        pos_feature = output["x2_glob_feature"]

        pos_moco = F.cosine_similarity(output["x1_query_pred"], output["x2_key_feature"], dim=-1)
        pos_moco_rev = F.cosine_similarity(output["x2_query_pred"], output["x1_key_feature"], dim=-1)

        query_pred = output["x1_query_pred"] # [B, D]
        tgt_key = output["x2_key_feature"] # [B, D]

        sim_matrix = query_pred @ tgt_key.T

        # positive and negative
        pos = sim_matrix.diag()
        mask = ~torch.eye(
            sim_matrix.size(0),
            dtype=torch.bool,
            device=sim_matrix.device,
        )
        neg = sim_matrix[mask]

        # top 1
        pred = sim_matrix.argmax(dim=1)
        target = torch.arange(
            sim_matrix.size(0),
            device=sim_matrix.device,
        )
        moco_top1 = (pred == target).float().mean()

        loss_12 = moco_contrastive_loss(output["x1_query_pred"], output["x2_key_feature"], temperature=0.2)
        loss_21 = moco_contrastive_loss(output["x2_query_pred"], output["x1_key_feature"], temperature=0.2)

        loss = loss_12 + loss_21 # moco_loss

        batch_size = anchor.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "positive_mean": pos.mean().item(),
        "negative_mean": neg.mean().item(),
        "positive_std": pos.std().item(),
        "negative_std": neg.std().item(),
        "gap": (pos.mean() - neg.mean()).item(),
        # "pos_cos": pos_moco,
        # "pos_cos_rev": pos_moco_rev,
        "top1" : moco_top1
    }


@torch.no_grad()
def validate_vit_moco_one_epoch(model, dataloader, device, topk=10):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        anchor_work_id = batch["src_work_id"].to(device)
        pos_work_id = batch["pos_work_id"].to(device)
        # neg_work_id = batch["neg_work_id"].to(device)

        anchor_perf_id = batch["src_label"].to(device)
        pos_perf_id = batch["pos_label"].to(device)
        # neg_perf_id = batch["neg_label"].to(device)

        anchor = batch["orig_inp"].to(device)
        positive = batch["pos_inp"].to(device)
        # negative = batch["neg_inp"].to(device)

        output = model(anchor, positive)

        anchor_feature = output["x1_glob_feature"]
        pos_feature = output["x2_glob_feature"]

        pos_moco = F.cosine_similarity(output["x1_query_pred"], output["x2_key_feature"], dim=-1)
        pos_moco_rev = F.cosine_similarity(output["x2_query_pred"], output["x1_key_feature"], dim=-1)

        query_pred = output["x1_query_pred"] # [B, D]
        tgt_key = output["x2_key_feature"] # [B, D]

        sim_matrix = query_pred @ tgt_key.T

        # positive and negative
        pos = sim_matrix.diag()
        mask = ~torch.eye(
            sim_matrix.size(0),
            dtype=torch.bool,
            device=sim_matrix.device,
        )
        neg = sim_matrix[mask]

        # top 1
        pred = sim_matrix.argmax(dim=1)
        target = torch.arange(
            sim_matrix.size(0),
            device=sim_matrix.device,
        )
        moco_top1 = (pred == target).float().mean()

        loss_12 = moco_contrastive_loss(output["x1_query_pred"], output["x2_key_feature"], temperature=0.2)
        loss_21 = moco_contrastive_loss(output["x2_query_pred"], output["x1_key_feature"], temperature=0.2)

        loss = loss_12 + loss_21 # moco_loss

        batch_size = anchor.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "positive_mean": pos.mean().item(),
        "negative_mean": neg.mean().item(),
        "positive_std": pos.std().item(),
        "negative_std": neg.std().item(),
        "gap": (pos.mean() - neg.mean()).item(),
        # "pos_cos": pos_moco,
        # "pos_cos_rev": pos_moco_rev,
        "top1" : moco_top1
    }


@torch.no_grad()
def evaluate_vit_moco_pn(model, dataloader, device, top_k=10):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        anchor_work_id = batch["src_work_id"].to(device)
        pos_work_id = batch["pos_work_id"].to(device)
        # neg_work_id = batch["neg_work_id"].to(device)

        anchor_perf_id = batch["src_label"].to(device)
        pos_perf_id = batch["pos_label"].to(device)
        # neg_perf_id = batch["neg_label"].to(device)

        anchor = batch["orig_inp"].to(device)
        positive = batch["pos_inp"].to(device)
        # negative = batch["neg_inp"].to(device)

        output = model(anchor, positive)

        anchor_feature = output["x1_glob_feature"]
        pos_feature = output["x2_glob_feature"]

        pos_moco = F.cosine_similarity(output["x1_query_pred"], output["x2_key_feature"], dim=-1)
        pos_moco_rev = F.cosine_similarity(output["x2_query_pred"], output["x1_key_feature"], dim=-1)

        query_pred = output["x1_query_pred"] # [B, D]
        tgt_key = output["x2_key_feature"] # [B, D]

        sim_matrix = query_pred @ tgt_key.T

        # positive and negative
        pos = sim_matrix.diag()
        mask = ~torch.eye(
            sim_matrix.size(0),
            dtype=torch.bool,
            device=sim_matrix.device,
        )
        neg = sim_matrix[mask]

        # top 1
        pred = sim_matrix.argmax(dim=1)
        target = torch.arange(
            sim_matrix.size(0),
            device=sim_matrix.device,
        )
        moco_top1 = (pred == target).float().mean()

        loss_12 = moco_contrastive_loss(output["x1_query_pred"], output["x2_key_feature"], temperature=0.2)
        loss_21 = moco_contrastive_loss(output["x2_query_pred"], output["x1_key_feature"], temperature=0.2)

        loss = loss_12 + loss_21 # moco_loss

        batch_size = anchor.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        

    return {
        "loss": total_loss / total_samples,
        "positive_mean": pos.mean().item(),
        "negative_mean": neg.mean().item(),
        "positive_std": pos.std().item(),
        "negative_std": neg.std().item(),
        "gap": (pos.mean() - neg.mean()).item(),
        # "pos_cos": pos_moco,
        # "pos_cos_rev": pos_moco_rev,
        "top1" : moco_top1
    }


def evaluate_vit_moco(model, dataloader, device, top_k=10):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    scores = []
    labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        anchor = batch["orig_inp"].to(device)
        positive = batch["cover_inp"].to(device)
        decision = batch["decision"].to(device)

        output = model(anchor, positive)
    
        s1 = F.cosine_similarity(
            output["x1_query_pred"],
            output["x2_key_feature"],
            dim=-1
        )

        s2 = F.cosine_similarity(
            output["x2_query_pred"],
            output["x1_key_feature"],
            dim=-1
        )

        score = 0.5 * (s1 + s2)

        scores.append(score.detach().cpu())
        labels.append(decision.detach().cpu())
        
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()

    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_idx = np.argmax(tpr - fpr)
    threshold = thresholds[best_idx]

    prediction = (scores >= threshold).astype(int)
    correct = (prediction == labels).sum() / len(prediction)
    auc = roc_auc_score(labels, scores)

    return {
        "threshold": threshold,
        "correct": correct,
        "auc": auc
    }


def log_train_val(log_path, epoch, train_metrics, val_metrics):
    log_line = (
        "Epoch {:03d} | "
        "Train Loss: {:.4f} | Pos: {:.4f} | Neg: {:.4f} | Pos Std: {:.4f} | Neg Std: {:.4f} | Gap: {:.4f} | Top1: {:.4f}\n"
        "Val Loss: {:.4f} | Pos: {:.4f} | Neg: {:.4f} | Pos Std: {:.4f} | Neg Std: {:.4f} | Gap: {:.4f} | Top1: {:.4f}\n"
    ).format(
        epoch,
        train_metrics["loss"],
        train_metrics["positive_mean"],
        train_metrics["negative_mean"],
        train_metrics["positive_std"],
        train_metrics["negative_std"],
        train_metrics["gap"],
        train_metrics["top1"],
        val_metrics["loss"],
        val_metrics["positive_mean"],
        val_metrics["negative_mean"],
        val_metrics["positive_std"],
        val_metrics["negative_std"],
        val_metrics["gap"],
        val_metrics["top1"],
    )

    print(log_line)

    with open(log_path, "a") as f:
        f.write(log_line + "\n")


def log_eval(log_path, dataset_name, metrics):
    log_line = (
        "{} | "
        "Loss: {:.4f} | Pos: {:.4f} | Neg: {:.4f} | Pos Std: {:.4f} | Neg Std: {:.4f} | Gap: {:.4f} | Top1: {:.4f}"
    ).format(
        dataset_name,
        metrics["loss"],
        metrics["positive_mean"],
        metrics["negative_mean"],
        metrics["positive_std"],
        metrics["negative_std"],
        metrics["gap"],
        metrics["top1"],
    )

    print(log_line)

    with open(log_path, "a") as f:
        f.write(log_line + "\n")


def log_ci_eval(log_path, dataset_name, metrics):
    log_line = (
        "{} | "
        "Threshold: {:.4f} | Accuracy: {:.4f} | AUC: {:.4f}"
    ).format(
        dataset_name,
        metrics["threshold"],
        metrics["correct"],
        metrics["auc"]
    )

    print(log_line)

    with open(log_path, "a") as f:
        f.write(log_line + "\n")