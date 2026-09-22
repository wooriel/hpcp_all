import torch

from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from models.f_losses import cosine_triplet_loss, cnn_retrieval_metrics, smooth_ap_loss


def train_cnn_one_epoch(model, dataloader, optimizer, device, margin=0.2, lambda_tri=1.0, lambda_ap=0.1):
    model.train()

    total_loss = 0.0
    total_tri = 0.0
    total_ap = 0.0
    total_samples = 0

    all_pos_sim = []
    all_neg_sim = []

    progress_bar = tqdm(dataloader, desc="Train", leave=False)

    for batch in progress_bar:
        anchor = batch["orig_inp"].to(device)
        positive = batch["pos_inp"].to(device)
        negative = batch["neg_inp"].to(device)

        anchor_work_id = batch["src_work_id"].to(device)
        pos_work_id = batch["pos_work_id"].to(device)
        neg_work_id = batch["neg_work_id"].to(device)

        optimizer.zero_grad()

        anchor_feature = model(anchor)
        pos_feature = model(positive)
        neg_feature = model(negative)

        tri_loss, cos_pos, cos_neg = cosine_triplet_loss(anchor_feature, pos_feature, neg_feature, margin=margin)

        embeddings = torch.cat([anchor_feature, pos_feature, neg_feature], dim=0)
        work_ids = torch.cat([anchor_work_id, pos_work_id, neg_work_id], dim=0)

        ap_loss = smooth_ap_loss(embeddings, work_ids, temperature=0.1)

        loss = lambda_tri * tri_loss + lambda_ap * ap_loss

        loss.backward()
        optimizer.step()

        batch_size = anchor.size(0)

        total_loss += loss.item() * batch_size
        total_tri += tri_loss.item() * batch_size
        total_ap += ap_loss.item() * batch_size
        total_samples += batch_size

        all_pos_sim.append(cos_pos.detach().cpu())
        all_neg_sim.append(cos_neg.detach().cpu())

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "tri": f"{tri_loss.item():.4f}",
            "ap": f"{ap_loss.item():.4f}",
        })

    pos_sim = torch.cat(all_pos_sim)
    neg_sim = torch.cat(all_neg_sim)

    pos_mean = pos_sim.mean().item()
    neg_mean = neg_sim.mean().item()

    scores = torch.cat([pos_sim, neg_sim]).numpy()
    labels = torch.cat([torch.ones_like(pos_sim), torch.zeros_like(neg_sim)]).numpy()

    auc = roc_auc_score(labels, scores)

    return {
        "loss": total_loss / total_samples,
        "triplet_loss": total_tri / total_samples,
        "ap_loss": total_ap / total_samples,
        "positive_mean": pos_mean,
        "negative_mean": neg_mean,
        "gap": pos_mean - neg_mean,
        "auc": auc,
    }


def validate_cnn(model, dataloader, device, margin=0.2, lambda_tri=1.0, lambda_ap=0.1, top_k=10):
    model.eval()

    total_loss = 0.0
    total_tri = 0.0
    total_ap = 0.0
    total_samples = 0

    all_pos_sim = []
    all_neg_sim = []

    all_embeddings = []
    all_work_ids = []
    all_perf_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            anchor = batch["orig_inp"].to(device)
            positive = batch["pos_inp"].to(device)
            negative = batch["neg_inp"].to(device)

            anchor_work_id = batch["src_work_id"].to(device)
            pos_work_id = batch["pos_work_id"].to(device)
            neg_work_id = batch["neg_work_id"].to(device)

            anchor_perf_id = batch["src_label"].to(device)
            pos_perf_id = batch["pos_label"].to(device)
            neg_perf_id = batch["neg_label"].to(device)

            anchor_feature = model(anchor)
            pos_feature = model(positive)
            neg_feature = model(negative)

            tri_loss, cos_pos, cos_neg = cosine_triplet_loss(anchor_feature, pos_feature, neg_feature, margin=margin)

            embeddings = torch.cat([anchor_feature, pos_feature, neg_feature], dim=0)
            work_ids = torch.cat([anchor_work_id, pos_work_id, neg_work_id], dim=0)

            ap_loss = smooth_ap_loss(embeddings, work_ids, temperature=0.1)
            loss = lambda_tri * tri_loss + lambda_ap * ap_loss

            batch_size = anchor.size(0)

            total_loss += loss.item() * batch_size
            total_tri += tri_loss.item() * batch_size
            total_ap += ap_loss.item() * batch_size
            total_samples += batch_size

            all_pos_sim.append(cos_pos.cpu())
            all_neg_sim.append(cos_neg.cpu())

            # whole-set retrieval embeddings
            all_embeddings.append(anchor_feature.cpu())
            all_embeddings.append(pos_feature.cpu())

            all_work_ids.append(anchor_work_id.cpu())
            all_work_ids.append(pos_work_id.cpu())

            all_perf_ids.append(anchor_perf_id.cpu())
            all_perf_ids.append(pos_perf_id.cpu())

    pos_sim = torch.cat(all_pos_sim)
    neg_sim = torch.cat(all_neg_sim)

    pos_mean = pos_sim.mean().item()
    neg_mean = neg_sim.mean().item()

    scores = torch.cat([pos_sim, neg_sim]).numpy()
    labels = torch.cat([torch.ones_like(pos_sim), torch.zeros_like(neg_sim)]).numpy()

    auc = roc_auc_score(labels, scores)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_work_ids = torch.cat(all_work_ids, dim=0)
    all_perf_ids = torch.cat(all_perf_ids, dim=0)

    ret = cnn_retrieval_metrics(all_embeddings, all_work_ids, all_perf_ids, top_k=top_k)

    return {
        "loss": total_loss / total_samples,
        "triplet_loss": total_tri / total_samples,
        "ap_loss": total_ap / total_samples,
        "positive_mean": pos_mean,
        "negative_mean": neg_mean,
        "gap": pos_mean - neg_mean,
        "auc": auc,
        "mrr": ret["mrr"],
        "map": ret["map"],
        "top1": ret["top1"],
        "topk": ret["topk"],
    }


def evaluate_cnn(model, dataloader, device, top_k=10):
    model.eval()

    all_embeddings = []
    all_work_ids = []
    all_perf_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation", leave=False):
            anchor = batch["orig_inp"].to(device)
            positive = batch["pos_inp"].to(device)

            anchor_feature = model(anchor)
            pos_feature = model(positive)

            all_embeddings.append(anchor_feature.cpu())
            all_embeddings.append(pos_feature.cpu())

            all_work_ids.append(batch["src_work_id"].cpu())
            all_work_ids.append(batch["pos_work_id"].cpu())

            all_perf_ids.append(batch["src_label"].cpu())
            all_perf_ids.append(batch["pos_label"].cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    work_ids = torch.cat(all_work_ids, dim=0)
    perf_ids = torch.cat(all_perf_ids, dim=0)

    metrics = cnn_retrieval_metrics(embeddings, work_ids, perf_ids, top_k=top_k)

    return metrics


def log_train_val(log_path, epoch, train_metrics, val_metrics):
    log_line = (
        "Epoch {:03d} | "
        "Train Loss: {:.4f} | Pos: {:.4f} | Neg: {:.4f} | Gap: {:.4f} | AUC: {:.4f}\n"
        "Val Loss: {:.4f} | Pos: {:.4f} | Neg: {:.4f} | Gap: {:.4f} | "
        "AUC: {:.4f} | Top1: {:.4f} | MRR: {:.4f} | MAP: {:.4f}"
    ).format(
        epoch,
        train_metrics["loss"],
        train_metrics["positive_mean"],
        train_metrics["negative_mean"],
        train_metrics["gap"],
        train_metrics["auc"],
        val_metrics["loss"],
        val_metrics["positive_mean"],
        val_metrics["negative_mean"],
        val_metrics["gap"],
        val_metrics["auc"],
        val_metrics["top1"],
        val_metrics["mrr"],
        val_metrics["map"],
    )

    print(log_line)

    with open(log_path, "a") as f:
        f.write(log_line + "\n")