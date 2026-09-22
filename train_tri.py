from tqdm import tqdm
from models.f_losses import triplet_loss, retrieval_loss, retrieval_metrics_val, diagonal_local_score
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def train_vit_triplet_one_epoch(model, dataloader, optimizer, device, glo_margin=0.2, loc_margin=0.2):
    model.train()
    total_loss = 0.0
    total_samples = 0
    lambda_tri = 1.0
    lambda_rank = 0.03
    lambda_local = 1.0

    all_pos_sim = []
    all_neg_sim = []

    total_tri = 0.0
    total_ret = 0.0
    total_loc = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        anchor_work_id = batch["src_work_id"].to(device)
        pos_work_id = batch["pos_work_id"].to(device)
        neg_work_id = batch["neg_work_id"].to(device)

        anchor_perf_id = batch["src_label"].to(device)
        pos_perf_id = batch["pos_label"].to(device)
        neg_perf_id = batch["neg_label"].to(device)

        anchor = batch["orig_inp"].to(device)
        positive = batch["pos_inp"].to(device)
        negative = batch["neg_inp"].to(device)

        optimizer.zero_grad()

        anchor_token = model.encode_tokens(anchor)
        pos_token = model.encode_tokens(positive)
        neg_token = model.encode_tokens(negative)

        anchor_feature = model.global_embedding(anchor_token)
        pos_feature = model.global_embedding(pos_token)
        neg_feature = model.global_embedding(neg_token)

        anchor_local = model.local_embedding(anchor_token)
        pos_local = model.local_embedding(pos_token)
        neg_local = model.local_embedding(neg_token)

        cos_pos = F.cosine_similarity(anchor_feature, pos_feature, dim=-1)
        cos_neg = F.cosine_similarity(anchor_feature, neg_feature, dim=-1)

        tri_loss = triplet_loss(cos_pos=cos_pos, cos_neg=cos_neg, margin=glo_margin)

        embeddings = torch.cat([anchor_feature, pos_feature, neg_feature], dim=0)
        work_ids = torch.cat([anchor_work_id, pos_work_id, neg_work_id], dim=0)
        perf_ids = torch.cat([anchor_perf_id, pos_perf_id, neg_perf_id], dim=0)

        rank_loss = retrieval_loss(embeddings=embeddings, work_ids=work_ids, perf_ids=perf_ids, temperature=0.1)
        global_loss = (lambda_tri * tri_loss + lambda_rank * rank_loss)

        pos_local_score = diagonal_local_score(anchor_local, pos_local, window=40)
        neg_local_score = diagonal_local_score(anchor_local, neg_local, window=40)
        local_loss = F.relu(neg_local_score - pos_local_score + loc_margin).mean()

        loss = (global_loss + lambda_local * local_loss)
        
        loss.backward()
        optimizer.step()

        batch_size = anchor.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_pos_sim.append(cos_pos.detach().cpu())
        all_neg_sim.append(cos_neg.detach().cpu())

        total_tri += tri_loss.item() * batch_size
        total_ret += rank_loss.item() * batch_size
        total_loc += local_loss.item() * batch_size

    pos_mean = torch.cat(all_pos_sim).mean().item()
    neg_mean = torch.cat(all_neg_sim).mean().item()

    return {
        "loss": total_loss / total_samples,
        "positive_cosine_mean": pos_mean,
        "negative_cosine_mean": neg_mean,
        "cosine_gap": pos_mean - neg_mean,
        "tri_loss": total_tri / total_samples,
        "rank_loss": total_ret / total_samples,
        "local_loss": total_loc / total_samples
    }


@torch.no_grad()
def validate_vit_triplet_one_epoch(model, dataloader, device, glo_margin=0.2, loc_margin=0.2, top_k=10):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    lambda_tri = 1.0
    lambda_rank = 0.03
    lambda_local = 1.0

    all_pos_sim = []
    all_neg_sim = []
    all_embeddings = [] # for MRR
    all_work_ids = []
    all_perf_ids = []

    total_tri = 0.0
    total_ret = 0.0
    total_loc = 0.0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    for batch in progress_bar:
        anchor_work_id = batch["src_work_id"].to(device)
        pos_work_id = batch["pos_work_id"].to(device)
        neg_work_id = batch["neg_work_id"].to(device)

        anchor_perf_id = batch["src_label"].to(device)
        pos_perf_id = batch["pos_label"].to(device)
        neg_perf_id = batch["neg_label"].to(device)

        anchor = batch["orig_inp"].to(device)
        positive = batch["pos_inp"].to(device)
        negative = batch["neg_inp"].to(device)

        anchor_token = model.encode_tokens(anchor)
        pos_token = model.encode_tokens(positive)
        neg_token = model.encode_tokens(negative)

        anchor_feature = model.global_embedding(anchor_token)
        pos_feature = model.global_embedding(pos_token)
        neg_feature = model.global_embedding(neg_token)

        anchor_local = model.local_embedding(anchor_token)
        pos_local = model.local_embedding(pos_token)
        neg_local = model.local_embedding(neg_token)

        # for all positive pairs
        src_work_id = batch["src_work_id"]
        cpos_work_id = batch["pos_work_id"]
        all_embeddings.append(anchor_feature.cpu())
        all_embeddings.append(pos_feature.cpu())
        assert torch.equal(src_work_id, cpos_work_id)
        all_work_ids.extend(src_work_id.cpu().tolist())
        all_work_ids.extend(cpos_work_id.cpu().tolist())
        all_perf_ids.extend(batch["src_label"].cpu().tolist())
        all_perf_ids.extend(batch["pos_label"].cpu().tolist())

        cos_pos = F.cosine_similarity(anchor_feature, pos_feature, dim=-1)
        cos_neg = F.cosine_similarity(anchor_feature, neg_feature, dim=-1)

        tri_loss = triplet_loss(cos_pos=cos_pos, cos_neg=cos_neg, margin=glo_margin)

        embeddings = torch.cat([anchor_feature, pos_feature, neg_feature], dim=0)
        work_ids = torch.cat([anchor_work_id, pos_work_id, neg_work_id], dim=0)
        perf_ids = torch.cat([anchor_perf_id, pos_perf_id, neg_perf_id], dim=0)

        rank_loss = retrieval_loss(embeddings=embeddings, work_ids=work_ids, perf_ids=perf_ids, temperature=0.1)
        global_loss = (lambda_tri * tri_loss + lambda_rank * rank_loss)

        pos_local_score = diagonal_local_score(anchor_local, pos_local, window=40)
        neg_local_score = diagonal_local_score(anchor_local, neg_local, window=40)
        local_loss = F.relu(neg_local_score - pos_local_score + loc_margin).mean()

        loss = (global_loss + lambda_local * local_loss)

        batch_size = anchor.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_pos_sim.append(cos_pos.detach().cpu())
        all_neg_sim.append(cos_neg.detach().cpu())

        total_tri += tri_loss.item() * batch_size
        total_ret += rank_loss.item() * batch_size
        total_loc += local_loss.item() * batch_size

    pos_sim = torch.cat(all_pos_sim)
    neg_sim = torch.cat(all_neg_sim)

    pos_mean = pos_sim.mean().item()
    neg_mean = neg_sim.mean().item()

    scores = torch.cat([pos_sim, neg_sim]).numpy()
    labels = torch.cat([torch.ones_like(pos_sim), torch.zeros_like(neg_sim)]).numpy()

    roc_auc = roc_auc_score(labels, scores)
    triplet_acc = (pos_sim > neg_sim).float().mean().item()
    margin_acc = (pos_sim >= neg_sim + glo_margin).float().mean().item()

    all_embeddings = torch.cat(all_embeddings, dim=0)
    ret_metrics = retrieval_metrics_val(all_embeddings, all_work_ids, all_perf_ids, top_k=top_k) # calculate MRR | MAP

    return {
        "loss": total_loss / total_samples,
        "positive_cosine_mean": pos_mean,
        "negative_cosine_mean": neg_mean,
        "cosine_gap": pos_mean - neg_mean,
        "roc_auc": roc_auc,
        "mrr": ret_metrics["mrr"],
        "map": ret_metrics["map"],
        "top1" : ret_metrics["top1"],
        "topk": ret_metrics["top{}".format(top_k)],
        "triplet_acc": triplet_acc,
        "margin_acc": margin_acc,
        "tri_loss": total_tri / total_samples,
        "rank_loss": total_ret / total_samples,
        "local_loss": total_loc / total_samples
    }


@torch.no_grad()
def evaluate_vit_triplet_one_epoch(model, dataloader, device, glo_margin=0.2, loc_margin=0.2, top_k=10):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    lambda_tri = 1.0
    lambda_rank = 0.03
    lambda_local = 1.0

    all_pos_sim = []
    all_neg_sim = []
    all_embeddings = [] # for MRR
    all_work_ids = []
    all_perf_ids = []
    
    for batch in tqdm(dataloader, desc="Evaluation", leave=False):
        anchor_work_id = batch["src_work_id"].to(device)
        pos_work_id = batch["pos_work_id"].to(device)
        neg_work_id = batch["neg_work_id"].to(device)

        anchor_perf_id = batch["src_label"].to(device)
        pos_perf_id = batch["pos_label"].to(device)
        neg_perf_id = batch["neg_label"].to(device)

        anchor = batch["orig_inp"].to(device)
        positive = batch["pos_inp"].to(device)
        negative = batch["neg_inp"].to(device)

        anchor_token = model.encode_tokens(anchor)
        pos_token = model.encode_tokens(positive)
        neg_token = model.encode_tokens(negative)

        anchor_feature = model.global_embedding(anchor_token)
        pos_feature = model.global_embedding(pos_token)
        neg_feature = model.global_embedding(neg_token)

        anchor_local = model.local_embedding(anchor_token)
        pos_local = model.local_embedding(pos_token)
        neg_local = model.local_embedding(neg_token)

        # for all positive pairs
        all_embeddings.append(anchor_feature)
        all_embeddings.append(pos_feature)
        assert torch.equal(batch["src_work_id"], batch["pos_work_id"])
        all_work_ids.extend(batch["src_work_id"].cpu().tolist())
        all_work_ids.extend(batch["pos_work_id"].tolist())
        all_perf_ids.extend(batch["src_label"].cpu().tolist())
        all_perf_ids.extend(batch["pos_label"].cpu().tolist())

        cos_pos = F.cosine_similarity(anchor_feature, pos_feature, dim=-1)
        cos_neg = F.cosine_similarity(anchor_feature, neg_feature, dim=-1)

        tri_loss = triplet_loss(cos_pos=cos_pos, cos_neg=cos_neg, margin=glo_margin)

        embeddings = torch.cat([anchor_feature, pos_feature, neg_feature], dim=0)
        work_ids = torch.cat([anchor_work_id, pos_work_id, neg_work_id], dim=0)
        perf_ids = torch.cat([anchor_perf_id, pos_perf_id, neg_perf_id], dim=0)

        rank_loss = retrieval_loss(embeddings=embeddings, work_ids=work_ids, perf_ids=perf_ids, temperature=0.1)
        global_loss = lambda_tri * tri_loss + lambda_rank * rank_loss

        pos_local_score = diagonal_local_score(anchor_local, pos_local, window=40)
        neg_local_score = diagonal_local_score(anchor_local, neg_local, window=40)
        local_loss = F.relu(neg_local_score - pos_local_score + loc_margin).mean()

        loss = global_loss + lambda_local * local_loss

        batch_size = anchor.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_pos_sim.append(cos_pos.detach().cpu())
        all_neg_sim.append(cos_neg.detach().cpu())

    pos_sim = torch.cat(all_pos_sim)
    neg_sim = torch.cat(all_neg_sim)

    pos_mean = pos_sim.mean().item()
    neg_mean = neg_sim.mean().item()

    scores = torch.cat([pos_sim, neg_sim]).numpy()
    labels = torch.cat([
        torch.ones_like(pos_sim),
        torch.zeros_like(neg_sim),
    ]).numpy()

    pos_std = pos_sim.std().item() if len(pos_sim) > 1 else float("nan")
    neg_std = neg_sim.std().item() if len(neg_sim) > 1 else float("nan")
    
    roc_auc = roc_auc_score(labels, scores)
    triplet_acc = (pos_sim > neg_sim).float().mean().item()
    margin_acc = (pos_sim >= neg_sim + glo_margin).float().mean().item()

    all_embeddings = torch.cat(all_embeddings, dim=0)
    ret_metrics = retrieval_metrics_val(all_embeddings, all_work_ids, all_perf_ids, top_k=top_k) # calculate MRR | MAP

    print("positive cosine:", pos_mean, "+/-", pos_std)
    print("negative cosine:", neg_mean, "+/-", neg_std)
    print("cosine gap:", pos_mean - neg_mean)
    print("pos cosine min/max:", pos_sim.min().item(), pos_sim.max().item())
    print("neg cosine min/max:", neg_sim.min().item(), neg_sim.max().item())
    print("ROC-AUC:", roc_auc)
    return {
        "loss": total_loss / total_samples,
        "positive_cosine_mean": pos_mean,
        "negative_cosine_mean": neg_mean,
        "positive_cosine_std": pos_std,
        "negative_cosine_std": neg_std,
        "cosine_gap": pos_mean - neg_mean,
        "roc_auc": roc_auc,
        "mrr": ret_metrics["mrr"],
        "map": ret_metrics["map"],
        "top1" : ret_metrics["top1"],
        "topk": ret_metrics["top{}".format(top_k)],
        "triplet_acc": triplet_acc,
        "margin_acc": margin_acc,
    }


def log_tri_epoch(log_path, epoch, train_metrics, val_metrics):
    log_line = (
        "Epoch {:03d} | "
        "Train Loss: {:.4f} | Pos: {:.4f} | Neg: {:.4f} | Gap: {:.4f} | "
        "Val Loss: {:.4f} | Pos: {:.4f} | Neg: {:.4f} | Gap: {:.4f} | "
        "AUC: {:.4f} | Triplet Acc: {:.4f} | Margin Acc: {:.4f} | "
        "MRR: {:.4f} | MAP: {:.4f} | Top1: {:.4f} | Topk: {:.4f} | "
        "Triplet Loss: {:.4f} | Rank Loss: {:.4f} | Local Loss: {:.4f}"
    ).format(
        epoch,
        train_metrics["loss"],
        train_metrics["positive_cosine_mean"],
        train_metrics["negative_cosine_mean"],
        train_metrics["cosine_gap"],
        val_metrics["loss"],
        val_metrics["positive_cosine_mean"],
        val_metrics["negative_cosine_mean"],
        val_metrics["cosine_gap"],
        val_metrics["roc_auc"],
        val_metrics["triplet_acc"],
        val_metrics["margin_acc"],
        val_metrics['mrr'],
        val_metrics['map'],
        val_metrics['top1'],
        val_metrics['topk'],
        val_metrics['tri_loss'],
        val_metrics['rank_loss'],
        val_metrics['local_loss']
    )

    print(log_line)

    with open(log_path, "a") as f:
        f.write(log_line + "\n")


# def log_tri_eval(log_path, dataset_name, metrics):
#     log_line = (
#         "{} | "
#         "Loss: {:.4f} | Pos: {:.4f} | Neg: {:.4f} | Gap: {:.4f} | "
#         "Pos Std: {:.4f} | Neg Std: {:.4f} | AUC: {:.4f} | "
#         "MRR: {:.4f} | MAP: {:.4f} | Top-1: {:.4f} | Top-k: {:.4f}"
#     ).format(
#         dataset_name,
#         metrics["loss"],
#         metrics["positive_cosine_mean"],
#         metrics["negative_cosine_mean"],
#         metrics["cosine_gap"],
#         metrics["positive_cosine_std"],
#         metrics["negative_cosine_std"],
#         metrics["roc_auc"],
#         metrics["mrr"],
#         metrics["map"],
#         metrics["top1"],
#         metrics["topk"],
#     )

#     print(log_line)

#     with open(log_path, "a") as f:
#         f.write(log_line + "\n")