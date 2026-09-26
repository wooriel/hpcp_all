import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def calculate_ci_metric(model, dataloader, device):
    all_cosine_sim = []
    all_labels = []

    for batch in dataloader:
        src = batch["orig_inp"].to(device)
        tgt = batch["cover_inp"].to(device)
        labels = batch["decision"].to(device)

        outputs = model(src, tgt)

        # global feature
        src_feature = outputs["src_feature"]
        tgt_feature = outputs["tgt_feature"]

        cosine_sim = F.cosine_similarity(
            src_feature,
            tgt_feature,
            dim=-1,
        )

        all_cosine_sim.append(cosine_sim.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_cosine_sim = torch.cat(all_cosine_sim, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    pos_mask = all_labels == 1
    neg_mask = all_labels == 0

    pos_sim = all_cosine_sim[pos_mask]
    neg_sim = all_cosine_sim[neg_mask]

    pos_mean = pos_sim.mean().item() if pos_sim.numel() > 0 else float("nan")
    neg_mean = neg_sim.mean().item() if neg_sim.numel() > 0 else float("nan")

    pos_std = pos_sim.std().item() if pos_sim.numel() > 1 else float("nan")
    neg_std = neg_sim.std().item() if neg_sim.numel() > 1 else float("nan")

    if pos_mask.any() and neg_mask.any():
        auc = roc_auc_score(
            all_labels.numpy(),
            all_cosine_sim.numpy(),
        )
    else:
        auc = float("nan")

    # scores = torch.cat(all_cosine_sim).numpy()
    # labels = torch.cat(all_labels).numpy()

    fpr, tpr, thresholds = roc_curve(all_labels, all_cosine_sim)

    j = tpr - fpr
    best_idx = j.argmax()

    best_threshold = thresholds[best_idx]

    print("Best threshold:", best_threshold)
    print("TPR:", tpr[best_idx])
    print("FPR:", fpr[best_idx])

    pred = (all_cosine_sim >= best_threshold).cpu().numpy().astype(int)

    accuracy = accuracy_score(all_labels, pred)
    precision = precision_score(all_labels, pred)
    recall = recall_score(all_labels, pred)
    f1 = f1_score(all_labels, pred)

    tn, fp, fn, tp = confusion_matrix(all_labels, pred).ravel()

    return {
        "positive_cosine_mean": pos_mean,
        "negative_cosine_mean": neg_mean,
        "positive_cosine_std": pos_std,
        "negative_cosine_std": neg_std,
        "cosine_gap": pos_mean - neg_mean,

        "roc_auc": auc,

        "scores": all_cosine_sim,
        "labels": all_labels,

        "threshold": best_threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def calculate_ci_moco_metric(model, dataloader, device):
    all_cosine_sim = []
    all_labels = []

    for batch in dataloader:
        src = batch["orig_inp"].to(device)
        tgt = batch["cover_inp"].to(device)
        labels = batch["decision"].to(device=device, dtype=torch.long)

        output = model(src, tgt)

        # MoCo query/key features
        q1 = F.normalize(output["x1_query_pred"], dim=-1)
        q2 = F.normalize(output["x2_query_pred"], dim=-1)

        k1 = F.normalize(output["x1_key_feature"], dim=-1)
        k2 = F.normalize(output["x2_key_feature"], dim=-1)

        # src -> tgt
        sim_12 = F.cosine_similarity(q1, k2, dim=-1)

        # tgt -> src
        sim_21 = F.cosine_similarity(q2, k1, dim=-1)

        # symmetric pair score
        cosine_sim = 0.5 * (sim_12 + sim_21)

        all_cosine_sim.append(cosine_sim.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_cosine_sim = torch.cat(all_cosine_sim, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # --------------------------------------------------
    # Positive / negative cosine statistics
    # --------------------------------------------------

    pos_mask = all_labels == 1
    neg_mask = all_labels == 0

    pos_sim = all_cosine_sim[pos_mask]
    neg_sim = all_cosine_sim[neg_mask]

    pos_mean = pos_sim.mean().item() if pos_sim.numel() > 0 else float("nan")
    neg_mean = neg_sim.mean().item() if neg_sim.numel() > 0 else float("nan")

    pos_std = pos_sim.std().item() if pos_sim.numel() > 1 else float("nan")
    neg_std = neg_sim.std().item() if neg_sim.numel() > 1 else float("nan")

    # Convert to NumPy once for sklearn
    scores_np = all_cosine_sim.numpy()
    labels_np = all_labels.numpy()

    # ROC-AUC
    if pos_mask.any() and neg_mask.any():
        auc = roc_auc_score(labels_np, scores_np)
    else:
        auc = float("nan")

    # Find threshold using Youden's J
    if pos_mask.any() and neg_mask.any():
        fpr, tpr, thresholds = roc_curve(labels_np, scores_np)

        j = tpr - fpr
        best_idx = j.argmax()

        best_threshold = thresholds[best_idx]
        best_tpr = tpr[best_idx]
        best_fpr = fpr[best_idx]
    else:
        best_threshold = float("nan")
        best_tpr = float("nan")
        best_fpr = float("nan")

    print("Best threshold:", best_threshold)
    print("TPR:", best_tpr)
    print("FPR:", best_fpr)

    # Binary prediction
    pred = (scores_np >= best_threshold).astype(int)

    accuracy = accuracy_score(labels_np, pred)
    precision = precision_score(labels_np, pred, zero_division=0)
    recall = recall_score(labels_np, pred, zero_division=0)
    f1 = f1_score(labels_np, pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        labels_np,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "positive_cosine_mean": pos_mean,
        "negative_cosine_mean": neg_mean,
        "positive_cosine_std": pos_std,
        "negative_cosine_std": neg_std,
        "cosine_gap": pos_mean - neg_mean,

        "roc_auc": auc,

        "scores": all_cosine_sim,
        "labels": all_labels,

        "threshold": best_threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,

        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def log_ci_eval(log_path, dataset_name, metrics):
    log_line = (
        "{} | "
        "Pos: {:.4f} | Neg: {:.4f} | Gap: {:.4f} | "
        "Pos Std: {:.4f} | Neg Std: {:.4f} | AUC: {:.4f} | "
        "Threshold: {:.4f} | Acc: {:.4f} | Prec: {:.4f} | Recall: {:.4f} | "
        "F1: {:.4f} | TP: {:.4f} | FP: {:.4f} | TN: {:.4f} | FN: {:.4f}"
    ).format(
        dataset_name,
        metrics["positive_cosine_mean"],
        metrics["negative_cosine_mean"],
        metrics["cosine_gap"],
        metrics["positive_cosine_std"],
        metrics["negative_cosine_std"],
        metrics["roc_auc"],
        metrics['threshold'],
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['tp'],
        metrics['fp'],
        metrics['tn'],
        metrics['fn'],
    )

    print(log_line)

    with open(log_path, "a") as f:
        f.write(log_line + "\n")