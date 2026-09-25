import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def cosine_pair_loss(cosine_similarity, pair_label, margin=0.3):
    pair_label = pair_label.long()

    pos_mask = pair_label == 1
    neg_mask = pair_label == 0

    if pos_mask.any():
        pos_loss = (
            1.0 - cosine_similarity[pos_mask]
        ).mean()
    else:
        pos_loss = cosine_similarity.new_tensor(0.0)

    if neg_mask.any():
        neg_loss = F.relu(
            cosine_similarity[neg_mask] - margin
        ).mean()
    else:
        neg_loss = cosine_similarity.new_tensor(0.0)

    loss = pos_loss + neg_loss

    return {
        "loss": loss,
        "positive_loss": pos_loss,
        "negative_loss": neg_loss,
    }


def contrastive_loss(src, tgt, label, margin=1.0):
    dist = F.pairwise_distance(src, tgt)
    pos_loss = label * dist.pow(2)
    neg_loss = (1 - label) * F.relu(margin - dist).pow(2)
    return (pos_loss + neg_loss).mean()


def triplet_loss(cos_pos, cos_neg, margin=0.1):
    return F.relu(cos_neg - cos_pos + margin).mean()


def retrieval_loss(embeddings, work_ids, perf_ids, temperature=0.1):
    embeddings = F.normalize(embeddings, dim=-1)

    work_ids = work_ids.view(-1)
    perf_ids = perf_ids.view(-1)

    sim = embeddings @ embeddings.T / temperature

    same_work = work_ids[:, None] == work_ids[None, :]
    same_perf = perf_ids[:, None] == perf_ids[None, :]

    # Positive = same work, but different performance
    pos_mask = same_work & ~same_perf

    # Do not compare a performance with another occurrence of itself
    valid_pair_mask = ~same_perf

    sim = sim.masked_fill(~valid_pair_mask, -float("inf"))

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

    valid_query = pos_mask.any(dim=1)

    pos_count = pos_mask.sum(dim=1).clamp_min(1)

    loss = -(
        log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1)
        / pos_count
    )

    if not valid_query.any():
        return embeddings.sum() * 0.0

    return loss[valid_query].mean()


# def retrieval_loss(embeddings, work_ids, temperature=0.1):
#     embeddings = F.normalize(embeddings, dim=-1)
#     sim = embeddings @ embeddings.T / temperature

#     work_ids = work_ids.view(-1)
#     pos_mask = work_ids[:, None] == work_ids[None, :]

#     self_mask = torch.eye(
#         len(work_ids),
#         dtype=torch.bool,
#         device=embeddings.device
#     )

#     pos_mask = pos_mask & ~self_mask

#     sim = sim.masked_fill(self_mask, -float("inf"))

#     log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

#     valid = pos_mask.any(dim=1)

#     loss = -(
#         (log_prob.masked_fill(~pos_mask, 0.0).sum(dim=1))
#         / pos_mask.sum(dim=1).clamp_min(1)
#     )

#     return loss[valid].mean()


def retrieval_metrics(embeddings, work_ids, top_k=10):
    similarity = embeddings @ embeddings.T

    work_ids = np.asarray(work_ids)

    reciprocal_ranks = []
    average_precisions = []
    top1_hits = []
    topk_hits = []

    for i in range(len(embeddings)):
        scores = similarity[i].clone()

        scores[i] = -float("inf") # exclude query itself

        relevant = work_ids == work_ids[i]
        relevant[i] = False

        if relevant.sum() == 0: # no relevant cover -> skip as query (I would filter using decision)
            continue

        ranking = torch.argsort(scores, descending=True).cpu().numpy()
        ranked_relevant = relevant[ranking]
        relevant_ranks = np.where(ranked_relevant)[0] + 1

        reciprocal_ranks.append(1.0 / relevant_ranks[0]) # MRR

        precision_sum = 0.0 # AP
        num_found = 0

        for rank, is_relevant in enumerate(ranked_relevant, start=1):
            if is_relevant:
                num_found += 1
                precision_sum += num_found / rank

        ap = precision_sum / relevant.sum()
        average_precisions.append(ap)

        top1_hits.append(float(ranked_relevant[0])) # Top-1
        topk_hits.append(float(ranked_relevant[:top_k].any())) # Top-k

    return {
        "mrr": np.mean(reciprocal_ranks),
        "map": np.mean(average_precisions),
        "top1": np.mean(top1_hits),
        "top{}".format(top_k): np.mean(topk_hits),
        "num_queries": len(reciprocal_ranks),
    }


def retrieval_metrics_val(embeddings, work_ids, performance_ids, top_k=10):
    if isinstance(embeddings, list):
        embeddings = torch.cat(embeddings, dim=0)

    work_ids = np.asarray(work_ids)
    performance_ids = np.asarray(performance_ids)

    assert len(embeddings) == len(work_ids) == len(performance_ids)

    # Keep each performance only once
    _, unique_indices = np.unique(performance_ids, return_index=True)
    unique_indices = np.sort(unique_indices)

    embeddings = embeddings[unique_indices]
    work_ids = work_ids[unique_indices]
    performance_ids = performance_ids[unique_indices]

    embeddings = F.normalize(embeddings, dim=-1)
    similarity = embeddings @ embeddings.T

    reciprocal_ranks = []
    average_precisions = []
    top1_hits = []
    topk_hits = []

    for i in range(len(embeddings)):
        scores = similarity[i].clone()
        scores[i] = -float("inf")

        relevant = (
            (work_ids == work_ids[i])
            & (performance_ids != performance_ids[i])
        )

        if relevant.sum() == 0:
            continue

        ranking = torch.argsort(scores, descending=True).cpu().numpy()

        ranked_relevant = relevant[ranking]
        relevant_ranks = np.where(ranked_relevant)[0] + 1

        reciprocal_ranks.append(1.0 / relevant_ranks[0])

        precision_sum = 0.0
        num_found = 0

        for rank, is_relevant in enumerate(ranked_relevant, start=1):
            if is_relevant:
                num_found += 1
                precision_sum += num_found / rank

        average_precisions.append(
            precision_sum / relevant.sum()
        )

        top1_hits.append(
            float(ranked_relevant[0])
        )

        topk_hits.append(
            float(ranked_relevant[:top_k].any())
        )

    return {
        "mrr": np.mean(reciprocal_ranks),
        "map": np.mean(average_precisions),
        "top1": np.mean(top1_hits),
        "topk": np.mean(topk_hits),
        "num_queries": len(reciprocal_ranks),
        "num_songs": len(embeddings),
    }


# retrieval metrics from global memory bank
def retrieval_metrics_from_sim(sim, positive_mask, candidate_mask, top_k=10):
    retrieval_sim = sim.masked_fill(~candidate_mask, float("-inf"))

    valid = positive_mask.any(dim=1)

    retrieval_sim = retrieval_sim[valid]
    positive_mask = positive_mask[valid]

    order = retrieval_sim.argsort(dim=1, descending=True) # ranking

    ranked_positive = torch.gather(positive_mask, dim=1, index=order)
    
    top1 = ranked_positive[:, 0].float().mean() # Top-1
    
    k = min(top_k, ranked_positive.size(1)) # Top-k
    topk = (ranked_positive[:, :k].any(dim=1).float().mean())
    
    ranks = torch.arange(1, ranked_positive.size(1) + 1, device=sim.device, dtype=torch.float32) # MRR
    reciprocal = ranked_positive.float() / ranks
    mrr = reciprocal.max(dim=1).values.mean()
    
    cumulative_hits = ranked_positive.float().cumsum(dim=1) # MAP
    precision_at_k = cumulative_hits / ranks
    num_positive = positive_mask.sum(dim=1).clamp_min(1)
    ap = (precision_at_k * ranked_positive.float()).sum(dim=1) / num_positive
    map_score = ap.mean()

    return {
        "top1": top1.item(),
        "topk": topk.item(),
        "mrr": mrr.item(),
        "map": map_score.item(),
    }


def local_alignment_score(src_tokens, tgt_tokens):
    src_tokens = F.normalize(src_tokens, dim=-1)
    tgt_tokens = F.normalize(tgt_tokens, dim=-1)

    sim = torch.bmm(src_tokens, tgt_tokens.transpose(1, 2))  # [B, Ns, Nt]

    best_tgt_per_src = sim.max(dim=2).values # [B, Ns] (maximum value)
    score = best_tgt_per_src.mean(dim=1) # [B]

    return score


def diagonal_local_score(src_tokens, tgt_tokens, window=8):
    src_tokens = F.normalize(src_tokens, dim=-1)
    tgt_tokens = F.normalize(tgt_tokens, dim=-1)

    sim = torch.bmm(
        src_tokens,
        tgt_tokens.transpose(1, 2)
    )  # [B, N, N]

    B, N, _ = sim.shape
    scores = []

    for offset in range(-N + 1, N):
        diag = sim.diagonal(offset=offset, dim1=1, dim2=2)

        if diag.shape[1] < window:
            continue

        local_avg = F.avg_pool1d(
            diag.unsqueeze(1),
            kernel_size=window,
            stride=1
        ).squeeze(1)

        scores.append(local_avg.max(dim=1).values)

    scores = torch.stack(scores, dim=1)

    return scores.max(dim=1).values


# def best_diagonal_window(sim, window=24): # sim: [B,N,M]
#     B, N, M = sim.shape

#     best_scores = []
#     best_src_starts = []
#     best_tgt_starts = []

#     for b in range(B):
#         best_score = -float("inf")
#         best_i = 0
#         best_j = 0

#         for offset in range(-N + 1, M):
#             diag = sim[b].diagonal(offset=offset)
#             if diag.numel() < window:
#                 continue

#             local_avg = F.avg_pool1d(diag.view(1, 1, -1), kernel_size=window, stride=1).view(-1)
#             value, idx = local_avg.max(dim=0)

#             if value > best_score:
#                 best_score = value

#                 if offset >= 0:
#                     src_start = idx.item()
#                     tgt_start = idx.item() + offset
#                 else:
#                     src_start = idx.item() - offset
#                     tgt_start = idx.item()

#                 best_i = src_start
#                 best_j = tgt_start

#         best_scores.append(best_score)
#         best_src_starts.append(best_i)
#         best_tgt_starts.append(best_j)

#     return (
#         torch.stack(best_scores),
#         torch.tensor(best_src_starts, device=sim.device),
#         torch.tensor(best_tgt_starts, device=sim.device),
#     )


def best_diagonal_window(sim, window=8, return_indices=False):
    B, N, M= sim.shape
    if window > N or window > M:
        raise ValueError(f"window={window} > N={N} or M={M}")
    
    sb, sn, sm = sim.stride()
    
    windows = sim.as_strided(
        size = (B,N-window+1,M-window+1,window),
        stride = (sb,sn,sm,sn+sm)
    )
    score_map = windows.mean(dim=-1)
    best_score, best_idx = score_map.flatten(1).max(dim=1)

    if not return_indices:
        return best_score, None, None
    
    n_tgt = M - window+1
    src_start = best_idx // n_tgt
    tgt_start = best_idx % n_tgt

    return best_score, src_start, tgt_start


def max_local_similarity(pair_sim): # pair_sim: [B, Nq, Nc]
    score = pair_sim.amax(dim=(1, 2))

    return score


# embedding related: MoCo3 contrastive loss
def moco_contrastive_loss(q, k, temperature=0.2):
    """_summary_: calculates cross entropy loss on similarity matrix

    Args:
        q (_type_): _description_
        k (_type_): _description_
        temperature (float, optional): _description_. Defaults to 0.2.

    Returns:
        _type_: _description_
    """
    logits = q @ k.T # [B, B]
    logits = logits / temperature

    # positive pair is same row index
    labels = torch.arange(
        q.size(0),
        device=q.device
    )

    return F.cross_entropy(logits, labels)


# def supervised_moco_loss(q, k, query_work_ids, key_work_ids, temperature=0.2):
#     # q, k assumed normalized
#     logits = q @ k.T / temperature   # [B, B]

#     # same work = positive
#     positive_mask = (query_work_ids[:, None] == key_work_ids[None, :])  # [B, B]

#     # log-softmax over all keys
#     log_prob = F.log_softmax(logits, dim=1)

#     # average over all positives for each query
#     positive_count = positive_mask.sum(dim=1)

#     valid = positive_count > 0

#     mean_log_prob_pos = (
#         (positive_mask.float() * log_prob).sum(dim=1)
#         / positive_count.clamp_min(1)
#     )

#     loss = -mean_log_prob_pos[valid].mean()

#     return loss


def supervised_moco_loss(q, k, work_ids, perf_ids, temperature=0.2):
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    logits = q @ k.T / temperature            # [2B, 2B]

    same_work = (work_ids[:, None] == work_ids[None, :])
    same_perf = ((perf_ids[:, None] == perf_ids[None, :]) & same_work)

    positive_mask = same_work & ~same_perf
    candidate_mask = ~same_perf

    # completely exclude exact same performance
    logits = logits.masked_fill(~candidate_mask, float("-inf"))

    log_prob = F.log_softmax(logits, dim=1)

    num_pos = positive_mask.sum(dim=1)
    valid = num_pos > 0

    # prevent 0 * -inf -> nan
    log_prob = torch.where(
        positive_mask,
        log_prob,
        torch.zeros_like(log_prob),
    )

    mean_log_prob_pos = (log_prob.sum(dim=1) / num_pos.clamp_min(1))

    return -mean_log_prob_pos[valid].mean()


def smooth_ap_loss(embeddings, work_ids, temperature=0.01):
    embeddings = F.normalize(embeddings, dim=-1)

    sim = embeddings @ embeddings.T
    B = sim.size(0)

    eye = torch.eye(B, dtype=torch.bool, device=sim.device)
    same_work = work_ids[:, None] == work_ids[None, :]

    positive_mask = same_work & ~eye
    candidate_mask = ~eye

    ap_list = []

    for i in range(B):
        scores = sim[i]
        positives = positive_mask[i]

        if positives.sum() == 0:
            continue

        valid_scores = scores[candidate_mask[i]]
        valid_pos = positives[candidate_mask[i]]

        pos_scores = valid_scores[valid_pos]
        precision_list = []

        for p_idx, s_pos in enumerate(pos_scores):
            score_diff = (valid_scores - s_pos) / temperature

            soft_rank = 1.0 + torch.sigmoid(score_diff).sum()

            pos_diff = (pos_scores - s_pos) / temperature

            pos_self_mask = torch.ones_like(pos_scores, dtype=torch.bool)
            pos_self_mask[p_idx] = False

            soft_pos_rank = 1.0 + torch.sigmoid(
                pos_diff[pos_self_mask]
            ).sum()

            precision = soft_pos_rank / soft_rank
            precision_list.append(precision)

        ap_i = torch.stack(precision_list).mean()
        ap_list.append(ap_i)

    return 1.0 - torch.stack(ap_list).mean()


# this version does not have process of calculating positive/candiate maskdef smooth_ap_from_sim(sim, positive_mask, candidate_mask, temperature=0.01):
def smooth_ap_from_sim(sim, positive_mask, candidate_mask, temperature=0.1):
    B = sim.size(0)

    ap_list = []

    for i in range(B):
        scores = sim[i]
        positives = positive_mask[i]

        if positives.sum() == 0:
            continue

        valid_scores = scores[candidate_mask[i]]
        valid_pos = positives[candidate_mask[i]]

        pos_scores = valid_scores[valid_pos]

        precision_list = []

        for p_idx, s_pos in enumerate(pos_scores):
            score_diff = (valid_scores - s_pos) / temperature

            soft_rank = 1.0 + torch.sigmoid(
                score_diff
            ).sum()

            pos_diff = (pos_scores - s_pos) / temperature

            pos_self_mask = torch.ones_like(
                pos_scores,
                dtype=torch.bool
            )

            pos_self_mask[p_idx] = False

            soft_pos_rank = 1.0 + torch.sigmoid(
                pos_diff[pos_self_mask]
            ).sum()

            precision = soft_pos_rank / soft_rank

            precision_list.append(precision)

        ap_i = torch.stack(precision_list).mean()
        ap_list.append(ap_i)

    if len(ap_list) == 0:
        return sim.sum() * 0.0

    return 1.0 - torch.stack(ap_list).mean()

## For CNN
def cosine_triplet_loss(anchor, positive, negative, margin=0.2):
    cos_pos = F.cosine_similarity(anchor, positive, dim=-1)
    cos_neg = F.cosine_similarity(anchor, negative, dim=-1)

    loss = F.relu(cos_neg - cos_pos + margin).mean()

    return loss, cos_pos, cos_neg


def cnn_retrieval_metrics(embeddings, work_ids, perf_ids, top_k=10):
    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=-1,
    )

    sim = embeddings @ embeddings.T

    same_work = (
        work_ids[:, None]
        == work_ids[None, :]
    )

    same_perf = (
        (
            perf_ids[:, None]
            == perf_ids[None, :]
        )
        & same_work
    )

    positive_mask = (
        same_work & ~same_perf
    )

    candidate_mask = ~same_perf

    valid = positive_mask.any(dim=1)

    sim = sim[valid]
    positive_mask = positive_mask[valid]
    candidate_mask = candidate_mask[valid]

    sim = sim.masked_fill(
        ~candidate_mask,
        float("-inf"),
    )

    order = sim.argsort(
        dim=1,
        descending=True,
    )

    ranked_positive = torch.gather(
        positive_mask,
        dim=1,
        index=order,
    )

    # Top1
    top1 = (
        ranked_positive[:, 0]
        .float()
        .mean()
    )

    # TopK
    k = min(
        top_k,
        ranked_positive.size(1),
    )

    topk = (
        ranked_positive[:, :k]
        .any(dim=1)
        .float()
        .mean()
    )

    # MRR
    ranks = torch.arange(
        1,
        ranked_positive.size(1) + 1,
        device=embeddings.device,
        dtype=torch.float32,
    )

    reciprocal_rank = (
        ranked_positive.float()
        / ranks
    ).max(dim=1).values

    mrr = reciprocal_rank.mean()

    # MAP
    cumulative_hits = (
        ranked_positive
        .float()
        .cumsum(dim=1)
    )

    precision_at_k = (
        cumulative_hits
        / ranks
    )

    num_positive = (
        positive_mask
        .sum(dim=1)
        .clamp_min(1)
    )

    ap = (
        precision_at_k
        * ranked_positive.float()
    ).sum(dim=1) / num_positive

    map_score = ap.mean()

    return {
        "top1": top1.item(),
        "topk": topk.item(),
        "mrr": mrr.item(),
        "map": map_score.item(),
    }


def retrieval_metrics_query(query_embeddings, candidate_embeddings, query_work_ids, candidate_work_ids, query_perf_ids=None, candidate_perf_ids=None, top_k=10):
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)

    sim = query_embeddings @ candidate_embeddings.T  # [Q, C]

    query_work_ids = torch.as_tensor(query_work_ids, device=sim.device)
    candidate_work_ids = torch.as_tensor(candidate_work_ids, device=sim.device)

    positive_mask = query_work_ids[:, None] == candidate_work_ids[None, :]

    if query_perf_ids is not None and candidate_perf_ids is not None:
        query_perf_ids = torch.as_tensor(query_perf_ids, device=sim.device)
        candidate_perf_ids = torch.as_tensor(candidate_perf_ids, device=sim.device)

        same_perf = (query_perf_ids[:, None] == candidate_perf_ids[None, :]) & positive_mask
        positive_mask = positive_mask & ~same_perf
        sim = sim.masked_fill(same_perf, float("-inf"))

    valid = positive_mask.any(dim=1)

    sim = sim[valid]
    positive_mask = positive_mask[valid]

    order = sim.argsort(dim=1, descending=True)
    ranked_positive = torch.gather(positive_mask, dim=1, index=order)

    top1 = ranked_positive[:, 0].float().mean()

    k = min(top_k, ranked_positive.shape[1])
    topk = ranked_positive[:, :k].any(dim=1).float().mean()

    ranks = torch.arange(1, ranked_positive.shape[1] + 1, device=sim.device, dtype=torch.float32)

    reciprocal_rank = (ranked_positive.float() / ranks).max(dim=1).values
    mrr = reciprocal_rank.mean()

    cumulative_hits = ranked_positive.float().cumsum(dim=1)
    precision_at_k = cumulative_hits / ranks
    num_positive = positive_mask.sum(dim=1).clamp_min(1)

    ap = (precision_at_k * ranked_positive.float()).sum(dim=1) / num_positive
    map_score = ap.mean()

    return {
        "mrr": mrr.item(),
        "map": map_score.item(),
        "top1": top1.item(),
        "topk": topk.item(),
    }


def retrieval_metrics_query_chunked(
    query_embeddings,
    candidate_embeddings,
    query_work_ids,
    candidate_work_ids,
    query_perf_ids,
    candidate_perf_ids,
    top_k=10,
    chunk_size=256,
):
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)

    query_work_ids = torch.as_tensor(query_work_ids)
    candidate_work_ids = torch.as_tensor(candidate_work_ids)
    query_perf_ids = torch.as_tensor(query_perf_ids)
    candidate_perf_ids = torch.as_tensor(candidate_perf_ids)

    rr_list = []
    ap_list = []
    top1_list = []
    topk_list = []

    for start in range(0, len(query_embeddings), chunk_size):
        end = min(start + chunk_size, len(query_embeddings))

        q = query_embeddings[start:end]
        sim = q @ candidate_embeddings.T

        q_work = query_work_ids[start:end]
        q_perf = query_perf_ids[start:end]

        positive_mask = q_work[:, None] == candidate_work_ids[None, :]
        same_perf = (q_perf[:, None] == candidate_perf_ids[None, :]) & positive_mask

        positive_mask = positive_mask & ~same_perf
        sim = sim.masked_fill(same_perf, float("-inf"))

        valid = positive_mask.any(dim=1)

        sim = sim[valid]
        positive_mask = positive_mask[valid]

        order = sim.argsort(dim=1, descending=True)
        ranked_positive = torch.gather(positive_mask, 1, order)

        top1_list.append(ranked_positive[:, 0].float())

        k = min(top_k, ranked_positive.shape[1])
        topk_list.append(ranked_positive[:, :k].any(dim=1).float())

        ranks = torch.arange(
            1,
            ranked_positive.shape[1] + 1,
            dtype=torch.float32,
        )

        rr = (ranked_positive.float() / ranks).max(dim=1).values
        rr_list.append(rr)

        cumulative_hits = ranked_positive.float().cumsum(dim=1)
        precision = cumulative_hits / ranks

        num_positive = positive_mask.sum(dim=1).clamp_min(1)
        ap = (precision * ranked_positive.float()).sum(dim=1) / num_positive
        ap_list.append(ap)

    return {
        "mrr": torch.cat(rr_list).mean().item(),
        "map": torch.cat(ap_list).mean().item(),
        "top1": torch.cat(top1_list).mean().item(),
        "topk": torch.cat(topk_list).mean().item(),
    }


@torch.no_grad()
def max_lag_similarity(model, src, tgt, crop_len=875, stride=175):
    # src: [B, 1, 96, T]
    # tgt: [B, 1, 96, T_tgt]

    B = src.shape[0]
    T = src.shape[-1]

    if T <= crop_len:
        outputs = model(src, tgt)
        score = outputs["cosine_similarity"]

        return score, torch.zeros(B, dtype=torch.long, device=src.device)

    best_score = torch.full((B,), -float("inf"), device=src.device)
    best_start = torch.zeros(B, dtype=torch.long, device=src.device)

    starts = list(range(0, T - crop_len + 1, stride))

    # Make sure the final part of the song is checked
    last_start = T - crop_len
    if starts[-1] != last_start:
        starts.append(last_start)

    for start in starts:
        query = src[..., start:start + crop_len]

        outputs = model(query, tgt)
        score = outputs["cosine_similarity"]  # [B]

        update = score > best_score

        best_score[update] = score[update]
        best_start[update] = start

    return best_score, best_start


@torch.no_grad()
def build_max_lag_score_matrix(model, dataloader, device, crop_len=875, stride=175):
    model.eval()

    pair_scores = []
    query_work_ids = []
    query_perf_ids = []
    candidate_work_ids = []
    candidate_perf_ids = []

    for batch in tqdm(dataloader, desc="Retrieval"):
        src = batch["orig_inp"].to(device) # full query song
        tgt = batch["cover_inp"].to(device) # candidate song

        score, best_lag = max_lag_similarity(
            model=model,
            src=src,
            tgt=tgt,
            crop_len=crop_len,
            stride=stride,
        )

        pair_scores.append(score.cpu())

        query_work_ids.extend(batch["work_id"].tolist())
        candidate_work_ids.extend(batch["work_id"].tolist())

        query_perf_ids.extend(batch["src_perf_id"].tolist())
        candidate_perf_ids.extend(batch["tgt_perf_id"].tolist())

    pair_scores = torch.cat(pair_scores, dim=0)

    return (
        pair_scores,
        query_work_ids,
        candidate_work_ids,
        query_perf_ids,
        candidate_perf_ids,
    )


# @torch.no_grad()
# def build_max_lag_score_matrix(model, query_loader, candidate_loader, device, crop_len=875, stride=175):
#     model.eval()

#     all_query_scores = []
#     query_work_ids = []
#     query_perf_ids = []

#     candidate_data = []
#     candidate_work_ids = []
#     candidate_perf_ids = []

#     # cache candidates first
#     for batch in candidate_loader:
#         candidate_data.append(batch["input"])
#         candidate_work_ids.extend(batch["work_id"].tolist())
#         candidate_perf_ids.extend(batch["perf_id"].tolist())

#     for qbatch in tqdm(query_loader, desc="Retrieval"):
#         query = qbatch["input"].to(device)

#         batch_scores = []

#         for cbatch in candidate_data:
#             candidate = cbatch.to(device)

#             score, _ = max_lag_similarity(
#                 model=model,
#                 src=query,
#                 tgt=candidate,
#                 crop_len=crop_len,
#                 stride=stride,
#             )

#             batch_scores.append(score.cpu())

#         batch_scores = torch.stack(batch_scores, dim=1)
#         all_query_scores.append(batch_scores)

#         query_work_ids.extend(qbatch["work_id"].tolist())
#         query_perf_ids.extend(qbatch["perf_id"].tolist())

#     score_matrix = torch.cat(all_query_scores, dim=0)

#     return (
#         score_matrix,
#         query_work_ids,
#         candidate_work_ids,
#         query_perf_ids,
#         candidate_perf_ids,
#     )


def retrieval_metrics_from_scores(
    scores,
    query_work_ids,
    candidate_work_ids,
    query_perf_ids=None,
    candidate_perf_ids=None,
    top_k=10,
):
    query_work_ids = torch.as_tensor(query_work_ids)
    candidate_work_ids = torch.as_tensor(candidate_work_ids)

    positive_mask = query_work_ids[:, None] == candidate_work_ids[None, :]

    if query_perf_ids is not None and candidate_perf_ids is not None:
        query_perf_ids = torch.as_tensor(query_perf_ids)
        candidate_perf_ids = torch.as_tensor(candidate_perf_ids)

        same_perf = (
            query_perf_ids[:, None] == candidate_perf_ids[None, :]
        ) & positive_mask

        positive_mask = positive_mask & ~same_perf
        scores = scores.masked_fill(same_perf, float("-inf"))

    valid = positive_mask.any(dim=1)

    scores = scores[valid]
    positive_mask = positive_mask[valid]

    order = scores.argsort(dim=1, descending=True)
    ranked_positive = torch.gather(positive_mask, 1, order)

    # Top-1
    top1 = ranked_positive[:, 0].float().mean()

    # Top-k
    k = min(top_k, ranked_positive.shape[1])
    topk = ranked_positive[:, :k].any(dim=1).float().mean()

    # MRR
    ranks = torch.arange(
        1,
        ranked_positive.shape[1] + 1,
        dtype=torch.float32,
    )

    rr = (ranked_positive.float() / ranks).max(dim=1).values
    mrr = rr.mean()

    # MAP
    cumulative_hits = ranked_positive.float().cumsum(dim=1)
    precision = cumulative_hits / ranks

    num_positive = positive_mask.sum(dim=1).clamp_min(1)

    ap = (
        precision * ranked_positive.float()
    ).sum(dim=1) / num_positive

    map_score = ap.mean()

    return {
        "mrr": mrr.item(),
        "map": map_score.item(),
        "top1": top1.item(),
        "topk": topk.item(),
    }
