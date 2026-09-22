from tqdm import tqdm
from models.f_losses import triplet_loss, retrieval_metrics_query_chunked, best_diagonal_window, smooth_ap_loss, max_lag_similarity, build_max_lag_score_matrix, retrieval_metrics_from_scores
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from gradient import gradient_cosine, get_loss_gradients, get_gradient_stats, gradient_norm, safe_mean


def train_vit_query_one_epoch(model, dataloader, optimizer, device, glo_margin=0.2, loc_margin=0.2, grad=False, top_k=10):
    model.train()
    total_loss = 0.0
    total_samples = 0

    lambda_tri = 1.0
    lambda_ap = 0.03
    lambda_local = 1.0

    all_pos_sim = []
    all_neg_sim = []
    
    # all_query_embeddings = []
    # all_cand_embeddings = []

    # all_query_work_ids = []
    # all_cand_work_ids = []

    # all_query_perf_ids = []
    # all_cand_perf_ids = []

    total_tri = 0.0
    total_ret = 0.0
    total_loc = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    if grad:
        grad_metrics = {
            "tokenizer_tri_norm": [],
            "tokenizer_rank_norm": [],
            "tokenizer_local_norm": [],

            "attn_tri_norm": [],
            "attn_rank_norm": [],
            "attn_local_norm": [],

            "pair_tri_norm": [],
            "pair_rank_norm": [],
            "pair_local_norm": [],

            "tokenizer_tri_rank_cos": [],
            "tokenizer_tri_local_cos": [],
            "tokenizer_rank_local_cos": [],

            "attn_tri_rank_cos": [],
            "attn_tri_local_cos": [],
            "attn_rank_local_cos": [],

            "pair_tri_rank_cos": [],
            "pair_tri_local_cos": [],
            "pair_rank_local_cos": [],

            "tokenizer_grad_norm": [],
            "attn_grad_norm": [],
            "pair_grad_norm": [],
        }

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

        outputp = model(anchor, positive)
        outputn = model(anchor, negative)

        anchor_feature = outputp["src_feature"]
        pos_feature = outputp["tgt_feature"]
        neg_feature = outputn["tgt_feature"]

        pos_sim = outputp["pair_similarity"]
        neg_sim = outputn["pair_similarity"]

        cos_pos = F.cosine_similarity(anchor_feature, pos_feature, dim=-1)
        cos_neg = F.cosine_similarity(anchor_feature, neg_feature, dim=-1)

        tri_loss = triplet_loss(cos_pos=cos_pos, cos_neg=cos_neg, margin=glo_margin)

        # query -> candidate
        # all_query_embeddings.append(anchor_feature.detach().cpu())
        # all_cand_embeddings.append(pos_feature.detach().cpu())

        # all_query_work_ids.extend(anchor_work_id.cpu().tolist())
        # all_cand_work_ids.extend(pos_work_id.cpu().tolist())

        # all_query_perf_ids.extend(anchor_perf_id.cpu().tolist())
        # all_cand_perf_ids.extend(pos_perf_id.cpu().tolist())

        # smooth-ap
        embeddings = torch.cat([anchor_feature, pos_feature, neg_feature], dim=0)
        work_ids = torch.cat([anchor_work_id, pos_work_id, neg_work_id], dim=0)

        ap_loss = smooth_ap_loss(embeddings, work_ids, temperature=0.1)
        global_loss = (lambda_tri * tri_loss + lambda_ap * ap_loss)

        pos_local_score, _, _ = best_diagonal_window(pos_sim, window=40)
        neg_local_score, _, _ = best_diagonal_window(neg_sim, window=40)

        local_loss = F.relu(neg_local_score - pos_local_score + loc_margin).mean()

        loss = (global_loss + lambda_local * local_loss)

        if grad:
            # individual loss: multiply with weight to see actual contribution
            tokenizer_tri_grad = get_loss_gradients(lambda_tri * tri_loss, model.tokenizer)
            tokenizer_rank_grad = get_loss_gradients(lambda_ap * ap_loss, model.tokenizer)
            tokenizer_local_grad = get_loss_gradients(lambda_local * local_loss, model.tokenizer)

            attn_tri_grad = get_loss_gradients(lambda_tri * tri_loss, model.self_attention)
            attn_rank_grad = get_loss_gradients(lambda_ap * ap_loss, model.self_attention)
            attn_local_grad = get_loss_gradients(lambda_local * local_loss, model.self_attention)

            pair_tri_grad = get_loss_gradients(lambda_tri * tri_loss, model.pair_projection)
            pair_rank_grad = get_loss_gradients(lambda_ap * ap_loss, model.pair_projection)
            pair_local_grad = get_loss_gradients(lambda_local * local_loss, model.pair_projection)

            # calculate norms
            tokenizer_tri_norm = gradient_norm(tokenizer_tri_grad)
            tokenizer_rank_norm = gradient_norm(tokenizer_rank_grad)
            tokenizer_local_norm = gradient_norm(tokenizer_local_grad)

            attn_tri_norm = gradient_norm(attn_tri_grad)
            attn_rank_norm = gradient_norm(attn_rank_grad)
            attn_local_norm = gradient_norm(attn_local_grad)

            pair_tri_norm = gradient_norm(pair_tri_grad)
            pair_rank_norm = gradient_norm(pair_rank_grad)
            pair_local_norm = gradient_norm(pair_local_grad)

            # cosine conflicts
            tokenizer_tri_rank_cos = gradient_cosine(tokenizer_tri_grad, tokenizer_rank_grad)
            tokenizer_tri_local_cos = gradient_cosine(tokenizer_tri_grad, tokenizer_local_grad)
            tokenizer_rank_local_cos = gradient_cosine(tokenizer_rank_grad, tokenizer_local_grad)

            attn_tri_rank_cos = gradient_cosine(attn_tri_grad, attn_rank_grad)
            attn_tri_local_cos = gradient_cosine(attn_tri_grad, attn_local_grad)
            attn_rank_local_cos = gradient_cosine(attn_rank_grad, attn_local_grad)

            pair_tri_rank_cos = gradient_cosine(pair_tri_grad, pair_rank_grad)
            pair_tri_local_cos = gradient_cosine(pair_tri_grad, pair_local_grad)
            pair_rank_local_cos = gradient_cosine(pair_rank_grad, pair_local_grad)

        loss.backward()

        if grad:
            tokenizer_grad = get_gradient_stats(model.tokenizer)
            attn_grad = get_gradient_stats(model.self_attention)
            pair_grad = get_gradient_stats(model.pair_projection)

            grad_metrics["tokenizer_tri_norm"].append(tokenizer_tri_norm)
            grad_metrics["tokenizer_rank_norm"].append(tokenizer_rank_norm)
            grad_metrics["tokenizer_local_norm"].append(tokenizer_local_norm)

            grad_metrics["attn_tri_norm"].append(attn_tri_norm)
            grad_metrics["attn_rank_norm"].append(attn_rank_norm)
            grad_metrics["attn_local_norm"].append(attn_local_norm)

            grad_metrics["pair_tri_norm"].append(pair_tri_norm)
            grad_metrics["pair_rank_norm"].append(pair_rank_norm)
            grad_metrics["pair_local_norm"].append(pair_local_norm)

            grad_metrics["tokenizer_tri_rank_cos"].append(tokenizer_tri_rank_cos)
            grad_metrics["tokenizer_tri_local_cos"].append(tokenizer_tri_local_cos)
            grad_metrics["tokenizer_rank_local_cos"].append(tokenizer_rank_local_cos)

            grad_metrics["attn_tri_rank_cos"].append(attn_tri_rank_cos)
            grad_metrics["attn_tri_local_cos"].append(attn_tri_local_cos)
            grad_metrics["attn_rank_local_cos"].append(attn_rank_local_cos)

            grad_metrics["pair_tri_rank_cos"].append(pair_tri_rank_cos)
            grad_metrics["pair_tri_local_cos"].append(pair_tri_local_cos)
            grad_metrics["pair_rank_local_cos"].append(pair_rank_local_cos)

            grad_metrics["tokenizer_grad_norm"].append(tokenizer_grad["grad_norm"])
            grad_metrics["attn_grad_norm"].append(attn_grad["grad_norm"])
            grad_metrics["pair_grad_norm"].append(pair_grad["grad_norm"])

        optimizer.step()

        batch_size = anchor.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_pos_sim.append(cos_pos.detach().cpu())
        all_neg_sim.append(cos_neg.detach().cpu())

        total_tri += tri_loss.item() * batch_size
        total_ret += ap_loss.item() * batch_size
        total_loc += local_loss.item() * batch_size

    pos_mean = torch.cat(all_pos_sim).mean().item()
    neg_mean = torch.cat(all_neg_sim).mean().item()

    pos_sim = torch.cat(all_pos_sim)
    neg_sim = torch.cat(all_neg_sim)

    pos_mean = pos_sim.mean().item()
    neg_mean = neg_sim.mean().item()

    scores = torch.cat([pos_sim, neg_sim]).numpy()
    labels = torch.cat([torch.ones_like(pos_sim), torch.zeros_like(neg_sim)]).numpy()

    roc_auc = roc_auc_score(labels, scores)

    # all_query_embeddings = torch.cat(all_query_embeddings, dim=0)
    # all_cand_embeddings = torch.cat(all_cand_embeddings, dim=0)

    # ret_metrics = retrieval_metrics_query(
    #     query_embeddings=all_query_embeddings,
    #     candidate_embeddings=all_cand_embeddings,
    #     query_work_ids=all_query_work_ids,
    #     candidate_work_ids=all_cand_work_ids,
    #     query_perf_ids=all_query_perf_ids,
    #     candidate_perf_ids=all_cand_perf_ids,
    #     top_k=top_k,
    # )

    if grad:
        return {
            "loss": total_loss / total_samples,
            "positive_cosine_mean": pos_mean,
            "negative_cosine_mean": neg_mean,
            "cosine_gap": pos_mean - neg_mean,
            "tri_loss": total_tri / total_samples,
            "ap_loss": total_ret / total_samples,
            "local_loss": total_loc / total_samples,

            "tokenizer_grad": {"grad_norm": safe_mean(grad_metrics["tokenizer_grad_norm"])},
            "attn_grad": {"grad_norm": safe_mean(grad_metrics["attn_grad_norm"])},
            "pair_grad": {"grad_norm": safe_mean(grad_metrics["pair_grad_norm"])},

            "gradient_conflict": {
                "tokenizer_tri_rank": safe_mean(grad_metrics["tokenizer_tri_rank_cos"]),
                "tokenizer_tri_local": safe_mean(grad_metrics["tokenizer_tri_local_cos"]),
                "tokenizer_rank_local": safe_mean(grad_metrics["tokenizer_rank_local_cos"]),

                "attn_tri_rank": safe_mean(grad_metrics["attn_tri_rank_cos"]),
                "attn_tri_local": safe_mean(grad_metrics["attn_tri_local_cos"]),
                "attn_rank_local": safe_mean(grad_metrics["attn_rank_local_cos"]),

                "pair_tri_rank": safe_mean(grad_metrics["pair_tri_rank_cos"]),
                "pair_tri_local": safe_mean(grad_metrics["pair_tri_local_cos"]),
                "pair_rank_local": safe_mean(grad_metrics["pair_rank_local_cos"])
            }
        }
    else:
        return {
            "loss": total_loss / total_samples,
            "positive_cosine_mean": pos_mean,
            "negative_cosine_mean": neg_mean,
            "cosine_gap": pos_mean - neg_mean,
            "roc_auc": roc_auc, # ratio that pos sim > neg sim
            # "mrr": ret_metrics["mrr"],
            # "map": ret_metrics["map"],
            # "top1" : ret_metrics["top1"],
            # "topk": ret_metrics["topk"],
            "tri_loss": total_tri / total_samples,
            "ap_loss": total_ret / total_samples,
            "local_loss": total_loc / total_samples
        }


@torch.no_grad()
def validate_vit_query_one_epoch(model, dataloader, device, glo_margin=0.2, loc_margin=0.2, top_k=10):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    lambda_tri = 1.0
    lambda_ap = 0.03
    lambda_local = 1.0

    all_pos_sim = []
    all_neg_sim = []
    
    all_query_embeddings = []
    all_cand_embeddings = []

    all_query_work_ids = []
    all_cand_work_ids = []

    all_query_perf_ids = []
    all_cand_perf_ids = []

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

        outputp = model(anchor, positive)
        outputn = model(anchor, negative)

        anchor_feature = outputp["src_feature"]
        pos_feature = outputp["tgt_feature"]
        neg_feature = outputn["tgt_feature"]

        pos_sim = outputp["pair_similarity"]
        neg_sim = outputn["pair_similarity"]

        cos_pos = F.cosine_similarity(anchor_feature, pos_feature, dim=-1)
        cos_neg = F.cosine_similarity(anchor_feature, neg_feature, dim=-1)

        tri_loss = triplet_loss(cos_pos=cos_pos, cos_neg=cos_neg, margin=glo_margin)

        # query -> candidate
        all_query_embeddings.append(anchor_feature.detach().cpu())
        all_cand_embeddings.append(pos_feature.detach().cpu())

        all_query_work_ids.extend(anchor_work_id.cpu().tolist())
        all_cand_work_ids.extend(pos_work_id.cpu().tolist())

        all_query_perf_ids.extend(anchor_perf_id.cpu().tolist())
        all_cand_perf_ids.extend(pos_perf_id.cpu().tolist())

        # smooth-ap
        embeddings = torch.cat([anchor_feature, pos_feature, neg_feature], dim=0)
        work_ids = torch.cat([anchor_work_id, pos_work_id, neg_work_id], dim=0)

        ap_loss = smooth_ap_loss(embeddings, work_ids, temperature=0.1)
        global_loss = (lambda_tri * tri_loss + lambda_ap * ap_loss)

        pos_local_score, _, _ = best_diagonal_window(pos_sim, window=40)
        neg_local_score, _, _ = best_diagonal_window(neg_sim, window=40)

        local_loss = F.relu(neg_local_score - pos_local_score + loc_margin).mean()

        loss = (global_loss + lambda_local * local_loss)

        batch_size = anchor.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_pos_sim.append(cos_pos.detach().cpu())
        all_neg_sim.append(cos_neg.detach().cpu())

        total_tri += tri_loss.item() * batch_size
        total_ret += ap_loss.item() * batch_size
        total_loc += local_loss.item() * batch_size

    pos_sim = torch.cat(all_pos_sim)
    neg_sim = torch.cat(all_neg_sim)

    pos_mean = pos_sim.mean().item()
    neg_mean = neg_sim.mean().item()

    scores = torch.cat([pos_sim, neg_sim]).numpy()
    labels = torch.cat([torch.ones_like(pos_sim), torch.zeros_like(neg_sim)]).numpy()

    roc_auc = roc_auc_score(labels, scores)

    all_query_embeddings = torch.cat(all_query_embeddings, dim=0)
    all_cand_embeddings = torch.cat(all_cand_embeddings, dim=0)

    ret_metrics = retrieval_metrics_query_chunked(
        query_embeddings=all_query_embeddings,
        candidate_embeddings=all_cand_embeddings,
        query_work_ids=all_query_work_ids,
        candidate_work_ids=all_cand_work_ids,
        query_perf_ids=all_query_perf_ids,
        candidate_perf_ids=all_cand_perf_ids,
        top_k=top_k,
    )

    return {
        "loss": total_loss / total_samples,
        "positive_cosine_mean": pos_mean,
        "negative_cosine_mean": neg_mean,
        "cosine_gap": pos_mean - neg_mean,
        "roc_auc": roc_auc, # ratio that pos sim > neg sim
        "mrr": ret_metrics["mrr"],
        "map": ret_metrics["map"],
        "top1" : ret_metrics["top1"],
        "topk": ret_metrics["topk"],
        "tri_loss": total_tri / total_samples,
        "ap_loss": total_ret / total_samples,
        "local_loss": total_loc / total_samples
    }


@torch.no_grad()
def evaluate_vit_query_one_epoch(model, dataloader, device, glo_margin=0.2, loc_margin=0.2, top_k=10):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    lambda_tri = 1.0
    lambda_ap = 0.03
    lambda_local = 1.0

    all_pos_sim = []
    all_neg_sim = []
    
    all_query_embeddings = []
    all_cand_embeddings = []

    all_query_work_ids = []
    all_cand_work_ids = []

    all_query_perf_ids = []
    all_cand_perf_ids = []
    
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

        outputp = model(anchor, positive)
        outputn = model(anchor, negative)

        anchor_feature = outputp["src_feature"]
        pos_feature = outputp["tgt_feature"]
        neg_feature = outputn["tgt_feature"]

        pos_sim = outputp["pair_similarity"]
        neg_sim = outputn["pair_similarity"]

        cos_pos = F.cosine_similarity(anchor_feature, pos_feature, dim=-1)
        cos_neg = F.cosine_similarity(anchor_feature, neg_feature, dim=-1)

        tri_loss = triplet_loss(cos_pos=cos_pos, cos_neg=cos_neg, margin=glo_margin)

        # query -> candidate
        all_query_embeddings.append(anchor_feature.detach().cpu())
        all_cand_embeddings.append(pos_feature.detach().cpu())

        all_query_work_ids.extend(anchor_work_id.cpu().tolist())
        all_cand_work_ids.extend(pos_work_id.cpu().tolist())

        all_query_perf_ids.extend(anchor_perf_id.cpu().tolist())
        all_cand_perf_ids.extend(pos_perf_id.cpu().tolist())

        # smooth-ap
        embeddings = torch.cat([anchor_feature, pos_feature, neg_feature], dim=0)
        work_ids = torch.cat([anchor_work_id, pos_work_id, neg_work_id], dim=0)

        ap_loss = smooth_ap_loss(embeddings, work_ids, temperature=0.1)
        global_loss = (lambda_tri * tri_loss + lambda_ap * ap_loss)

        pos_local_score, _, _ = best_diagonal_window(pos_sim, window=40)
        neg_local_score, _, _ = best_diagonal_window(neg_sim, window=40)

        local_loss = F.relu(neg_local_score - pos_local_score + loc_margin).mean()

        loss = (global_loss + lambda_local * local_loss)

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

    all_query_embeddings = torch.cat(all_query_embeddings, dim=0)
    all_cand_embeddings = torch.cat(all_cand_embeddings, dim=0)

    ret_metrics = retrieval_metrics_query_chunked(
        query_embeddings=all_query_embeddings,
        candidate_embeddings=all_cand_embeddings,
        query_work_ids=all_query_work_ids,
        candidate_work_ids=all_cand_work_ids,
        query_perf_ids=all_query_perf_ids,
        candidate_perf_ids=all_cand_perf_ids,
        top_k=top_k,
    )

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
        "topk": ret_metrics["topk"],
    }


def log_train_val(log_path, epoch, train_metrics, val_metrics):
    log_line = (
        "Epoch {:03d} | "
        "Train Loss: {:.4f} | Pos: {:.4f} | Neg: {:.4f} | Gap: {:.4f} | "
        "AUC: {:.4f} | "
        "Triplet Loss: {:.4f} | AP Loss: {:.4f} | Local Loss: {:.4f}\n"
        "Val Loss: {:.4f} | Pos: {:.4f} | Neg: {:.4f} | Gap: {:.4f}\n"
        "AUC: {:.4f} | MRR: {:.4f} | MAP: {:.4f} | Top1: {:.4f} | Topk: {:.4f} | "
        "Triplet Loss: {:.4f} | AP Loss: {:.4f} | Local Loss: {:.4f}"
    ).format(
        epoch,
        train_metrics["loss"],
        train_metrics["positive_cosine_mean"],
        train_metrics["negative_cosine_mean"],
        train_metrics["cosine_gap"],
        train_metrics["roc_auc"],
        # train_metrics['mrr'],
        # train_metrics['map'],
        # train_metrics['top1'],
        # train_metrics['topk'],
        train_metrics['tri_loss'],
        train_metrics['ap_loss'],
        train_metrics['local_loss'],
        val_metrics["loss"],
        val_metrics["positive_cosine_mean"],
        val_metrics["negative_cosine_mean"],
        val_metrics["cosine_gap"],
        val_metrics["roc_auc"],
        val_metrics['mrr'],
        val_metrics['map'],
        val_metrics['top1'],
        val_metrics['topk'],
        val_metrics['tri_loss'],
        val_metrics['ap_loss'],
        val_metrics['local_loss']
    )

    print(log_line)

    with open(log_path, "a") as f:
        f.write(log_line + "\n")


def log_eval(log_path, dataset_name, metrics):
    log_line = (
        "{} | "
        "Loss: {:.4f} | Pos: {:.4f} | Neg: {:.4f} | Gap: {:.4f} | "
        "Pos Std: {:.4f} | Neg Std: {:.4f} | AUC: {:.4f} | "
        "MRR: {:.4f} | MAP: {:.4f} | Top-1: {:.4f} | Top-k: {:.4f}"
    ).format(
        dataset_name,
        metrics["loss"],
        metrics["positive_cosine_mean"],
        metrics["negative_cosine_mean"],
        metrics["cosine_gap"],
        metrics["positive_cosine_std"],
        metrics["negative_cosine_std"],
        metrics["roc_auc"],
        metrics['mrr'],
        metrics['map'],
        metrics['top1'],
        metrics['topk'],
    )

    print(log_line)

    with open(log_path, "a") as f:
        f.write(log_line + "\n")


def log_eval2(log_path, dataset_name, metrics, metrics2):
    log_line = (
        "{} | "
        "Pos: {:.4f} | Neg: {:.4f} | Gap: {:.4f} | "
        "Pos Std: {:.4f} | Neg Std: {:.4f} | AUC: {:.4f} | "
        "MRR: {:.4f} | MAP: {:.4f} | Top-1: {:.4f} | Top-k: {:.4f}"
    ).format(
        dataset_name,
        # metrics["loss"],
        metrics["positive_cosine_mean"],
        metrics["negative_cosine_mean"],
        metrics["cosine_gap"],
        metrics["positive_cosine_std"],
        metrics["negative_cosine_std"],
        metrics["roc_auc"],
        metrics2['mrr'],
        metrics2['map'],
        metrics2['top1'],
        metrics2['topk'],
    )

    print(log_line)

    with open(log_path, "a") as f:
        f.write(log_line + "\n")


@torch.no_grad()
def evaluate_vit_max_lag(model, dataloader, device, crop_len=875, stride=175):
    model.eval()

    all_scores = []
    all_labels = []
    all_best_lags = []

    for batch in tqdm(dataloader, desc="Evaluation", leave=False):
        src = batch["orig_inp"].to(device)
        tgt = batch["cover_inp"].to(device)

        pair_label = batch["decision"].to(device=device, dtype=torch.long)

        max_score, best_lag = max_lag_similarity(
            model=model,
            src=src,
            tgt=tgt,
            crop_len=crop_len,
            stride=stride,
        )

        all_scores.append(max_score.cpu())
        all_labels.append(pair_label.cpu())
        all_best_lags.append(best_lag.cpu())

    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)
    all_best_lags = torch.cat(all_best_lags)

    pos_mask = all_labels == 1
    neg_mask = all_labels == 0

    pos_sim = all_scores[pos_mask]
    neg_sim = all_scores[neg_mask]

    pos_mean = pos_sim.mean().item() if pos_mask.any() else float("nan")
    neg_mean = neg_sim.mean().item() if neg_mask.any() else float("nan")

    pos_std = pos_sim.std().item() if len(pos_sim) > 1 else float("nan")
    neg_std = neg_sim.std().item() if len(neg_sim) > 1 else float("nan")

    if pos_mask.any() and neg_mask.any():
        auc = roc_auc_score(all_labels.numpy(), all_scores.numpy())
    else:
        auc = float("nan")

    print("positive cosine:", pos_mean, "+/-", pos_std)
    print("negative cosine:", neg_mean, "+/-", neg_std)
    print("cosine gap:", pos_mean - neg_mean)
    print("score min/max:", all_scores.min().item(), all_scores.max().item())
    print("ROC-AUC:", auc)

    return {
        "positive_cosine_mean": pos_mean,
        "negative_cosine_mean": neg_mean,
        "positive_cosine_std": pos_std,
        "negative_cosine_std": neg_std,
        "cosine_gap": pos_mean - neg_mean,
        "roc_auc": auc,
        "scores": all_scores.mean().item(),
        "best_lags": all_best_lags.float().mean().item(),
    }


@torch.no_grad()
def evaluate_retrieval_max_lag(model, dataloader, device, crop_len=875, stride=175, top_k=10):
    score_matrix, query_work_ids, candidate_work_ids, query_perf_ids, candidate_perf_ids = build_max_lag_score_matrix(
        model=model,
        dataloader=dataloader,
        device=device,
        crop_len=crop_len,
        stride=stride,
    )

    ret_metrics = retrieval_metrics_from_scores(
        scores=score_matrix,
        query_work_ids=query_work_ids,
        candidate_work_ids=candidate_work_ids,
        query_perf_ids=query_perf_ids,
        candidate_perf_ids=candidate_perf_ids,
        top_k=top_k,
    )

    return ret_metrics