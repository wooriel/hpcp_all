from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def evaluate_embedding_similarity_vit(model, dataloader, device):
    model.eval()
    all_vit_cos = []
    all_selfattn_cos = []
    all_final_cos = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluation", leave=False):
        src = batch["orig_inp"].to(device)
        tgt = batch["pos_inp"].to(device)
        pair_label = batch["src_wid"].to(device=device, dtype=torch.long)
        B = pair_label.size(0)

        # 1. ViT tokenizer output
        zvit_src = model.tokenizer(src)   # [B*head, N, D]
        zvit_tgt = model.tokenizer(tgt)

        # pool tokens manually
        zvit_src = zvit_src.mean(dim=1)   # [B*head, D]
        zvit_tgt = zvit_tgt.mean(dim=1)

        # combine heads
        zvit_src = zvit_src.reshape(B, model.head, model.dim).mean(dim=1)
        zvit_tgt = zvit_tgt.reshape(B, model.head, model.dim).mean(dim=1)

        vit_cos_sim = F.cosine_similarity(zvit_src, zvit_tgt, dim=-1)

        # 2. ViT + self-attention + actual song_pool
        z_src = model.tokenizer(src)
        z_tgt = model.tokenizer(tgt)

        z_src = model.self_attention(z_src)
        z_tgt = model.self_attention(z_tgt)

        z_src = model.song_pool(z_src)   # [B*head, D]
        z_tgt = model.song_pool(z_tgt)

        z_src = z_src.reshape(B, model.head, model.dim).mean(dim=1)
        z_tgt = z_tgt.reshape(B, model.head, model.dim).mean(dim=1)

        selfattn_cos_sim = F.cosine_similarity(z_src, z_tgt, dim=-1)

        # 3. Final embedding
        final_src = model.encode(src)    # [B, D]
        final_tgt = model.encode(tgt)

        final_cos_sim = F.cosine_similarity(final_src, final_tgt, dim=-1)

        all_vit_cos.append(vit_cos_sim.cpu())
        all_selfattn_cos.append(selfattn_cos_sim.cpu())
        all_final_cos.append(final_cos_sim.cpu())
        all_labels.append(pair_label.cpu())

    all_vit_cos = torch.cat(all_vit_cos)
    all_selfattn_cos = torch.cat(all_selfattn_cos)
    all_final_cos = torch.cat(all_final_cos)
    all_labels = torch.cat(all_labels)

    pos_mask = all_labels == 1
    neg_mask = all_labels == 0

    def print_similarity_stats(name, scores):
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        pos_mean = pos_scores.mean().item()
        neg_mean = neg_scores.mean().item()
        gap = pos_mean - neg_mean

        auc = roc_auc_score(all_labels.numpy(), scores.numpy()) if pos_mask.any() and neg_mask.any() else float("nan")

        print(f"\n[{name}]")
        print("positive:", pos_mean, "+/-", pos_scores.std().item())
        print("negative:", neg_mean, "+/-", neg_scores.std().item())
        print("gap:", gap)
        print("ROC-AUC:", auc)

        return {"positive_mean": pos_mean, "negative_mean": neg_mean, "gap": gap, "auc": auc}

    vit_stats = print_similarity_stats("ViT tokenizer embedding", all_vit_cos)
    selfattn_stats = print_similarity_stats("Self-attention embedding", all_selfattn_cos)
    final_stats = print_similarity_stats("Final embedding", all_final_cos)

    return {
        "vit_positive_mean": vit_stats["positive_mean"],
        "vit_negative_mean": vit_stats["negative_mean"],
        "vit_gap": vit_stats["gap"],
        "vit_auc": vit_stats["auc"],
        "selfattn_positive_mean": selfattn_stats["positive_mean"],
        "selfattn_negative_mean": selfattn_stats["negative_mean"],
        "selfattn_gap": selfattn_stats["gap"],
        "selfattn_auc": selfattn_stats["auc"],
        "final_positive_mean": final_stats["positive_mean"],
        "final_negative_mean": final_stats["negative_mean"],
        "final_gap": final_stats["gap"],
        "final_auc": final_stats["auc"],
    }


def similarity_stats(pos_scores, neg_scores):
    pos_mean = pos_scores.mean().item()
    neg_mean = neg_scores.mean().item()

    return {
        "positive_mean": pos_mean,
        "negative_mean": neg_mean,
        "positive_std": pos_scores.std().item(),
        "negative_std": neg_scores.std().item(),
        "gap": pos_mean - neg_mean,
        "positive_min": pos_scores.min().item(),
        "positive_max": pos_scores.max().item(),
        "negative_min": neg_scores.min().item(),
        "negative_max": neg_scores.max().item(),
    }


@torch.no_grad()
def evaluate_embedding_hpcp_vit(model, dataloader, device):
    model.eval()

    all_tokenizer_pos_cos = []
    all_tokenizer_neg_cos = []

    all_global_pos_cos = []
    all_global_neg_cos = []

    all_local_pos_sim = []
    all_local_neg_sim = []

    all_pair_pos_sim = []
    all_pair_neg_sim = []

    for batch in tqdm(dataloader, desc="Embedding Evaluation", leave=False):

        src = batch["orig_inp"].to(device)
        pos = batch["pos_inp"].to(device)
        neg = batch["neg_inp"].to(device)

        B = src.shape[0]

        # ============================================================
        # 1. Tokenizer
        # ============================================================

        src_tok_raw = model.tokenizer(src)
        pos_tok_raw = model.tokenizer(pos)
        neg_tok_raw = model.tokenizer(neg)

        src_tokenizer_emb = src_tok_raw.mean(dim=1)
        pos_tokenizer_emb = pos_tok_raw.mean(dim=1)
        neg_tokenizer_emb = neg_tok_raw.mean(dim=1)

        src_tokenizer_emb = src_tokenizer_emb.reshape(
            B, model.head, model.dim
        ).mean(dim=1)

        pos_tokenizer_emb = pos_tokenizer_emb.reshape(
            B, model.head, model.dim
        ).mean(dim=1)

        neg_tokenizer_emb = neg_tokenizer_emb.reshape(
            B, model.head, model.dim
        ).mean(dim=1)

        tokenizer_pos_cos = F.cosine_similarity(
            src_tokenizer_emb,
            pos_tokenizer_emb,
            dim=-1
        )

        tokenizer_neg_cos = F.cosine_similarity(
            src_tokenizer_emb,
            neg_tokenizer_emb,
            dim=-1
        )

        # ============================================================
        # 2. Self-attention
        # ============================================================

        src_tokens = model.self_attention(src_tok_raw)
        pos_tokens = model.self_attention(pos_tok_raw)
        neg_tokens = model.self_attention(neg_tok_raw)

        src_tokens = src_tokens.reshape(
            B, model.head, src_tokens.shape[1], model.dim
        )

        pos_tokens = pos_tokens.reshape(
            B, model.head, pos_tokens.shape[1], model.dim
        )

        neg_tokens = neg_tokens.reshape(
            B, model.head, neg_tokens.shape[1], model.dim
        )

        # ============================================================
        # 3. Global
        # ============================================================

        src_global = model.global_embedding(src_tokens)
        pos_global = model.global_embedding(pos_tokens)
        neg_global = model.global_embedding(neg_tokens)

        global_pos_cos = F.cosine_similarity(
            src_global,
            pos_global,
            dim=-1
        )

        global_neg_cos = F.cosine_similarity(
            src_global,
            neg_global,
            dim=-1
        )

        # ============================================================
        # 4. Local embedding
        # ============================================================

        src_local = model.local_embedding(src_tokens)
        pos_local = model.local_embedding(pos_tokens)
        neg_local = model.local_embedding(neg_tokens)

        # Raw local similarity
        local_pos_matrix = torch.bmm(
            src_local,
            pos_local.transpose(1, 2)
        )

        local_neg_matrix = torch.bmm(
            src_local,
            neg_local.transpose(1, 2)
        )

        # Best target match for each source token
        local_pos_sim = local_pos_matrix.max(dim=2).values.mean(dim=1)
        local_neg_sim = local_neg_matrix.max(dim=2).values.mean(dim=1)


        # ============================================================
        # 5. Pair projection
        # ============================================================

        _, _, pair_pos_sim = model.pair_projection(
            src_local,
            pos_local
        )

        _, _, pair_neg_sim = model.pair_projection(
            src_local,
            neg_local
        )

        if pair_pos_sim.ndim > 1 and pair_pos_sim.shape[-1] == 1:
            pair_pos_sim = pair_pos_sim.squeeze(-1)

        if pair_neg_sim.ndim > 1 and pair_neg_sim.shape[-1] == 1:
            pair_neg_sim = pair_neg_sim.squeeze(-1)

        # ============================================================
        # Store
        # ============================================================

        all_tokenizer_pos_cos.append(tokenizer_pos_cos.cpu())
        all_tokenizer_neg_cos.append(tokenizer_neg_cos.cpu())

        all_global_pos_cos.append(global_pos_cos.cpu())
        all_global_neg_cos.append(global_neg_cos.cpu())

        all_local_pos_sim.append(local_pos_sim.cpu())
        all_local_neg_sim.append(local_neg_sim.cpu())

        all_pair_pos_sim.append(pair_pos_sim.cpu())
        all_pair_neg_sim.append(pair_neg_sim.cpu())

    # ================================================================
    # Concatenate
    # ================================================================

    all_tokenizer_pos_cos = torch.cat(all_tokenizer_pos_cos)
    all_tokenizer_neg_cos = torch.cat(all_tokenizer_neg_cos)

    all_global_pos_cos = torch.cat(all_global_pos_cos)
    all_global_neg_cos = torch.cat(all_global_neg_cos)

    all_local_pos_sim = torch.cat(all_local_pos_sim)
    all_local_neg_sim = torch.cat(all_local_neg_sim)

    all_pair_pos_sim = torch.cat(all_pair_pos_sim)
    all_pair_neg_sim = torch.cat(all_pair_neg_sim)

    # ================================================================
    # Statistics
    # ================================================================

    tokenizer_stats = similarity_stats(
        all_tokenizer_pos_cos,
        all_tokenizer_neg_cos
    )

    global_stats = similarity_stats(
        all_global_pos_cos,
        all_global_neg_cos
    )

    local_stats = similarity_stats(
        all_local_pos_sim,
        all_local_neg_sim
    )

    pair_stats = similarity_stats(
        all_pair_pos_sim,
        all_pair_neg_sim
    )

    # ================================================================
    # Return
    # ================================================================

    return {
        "tokenizer_positive_mean": tokenizer_stats["positive_mean"],
        "tokenizer_negative_mean": tokenizer_stats["negative_mean"],
        "tokenizer_positive_std": tokenizer_stats["positive_std"],
        "tokenizer_negative_std": tokenizer_stats["negative_std"],
        "tokenizer_gap": tokenizer_stats["gap"],

        "global_positive_mean": global_stats["positive_mean"],
        "global_negative_mean": global_stats["negative_mean"],
        "global_positive_std": global_stats["positive_std"],
        "global_negative_std": global_stats["negative_std"],
        "global_gap": global_stats["gap"],

        "local_positive_mean": local_stats["positive_mean"],
        "local_negative_mean": local_stats["negative_mean"],
        "local_positive_std": local_stats["positive_std"],
        "local_negative_std": local_stats["negative_std"],
        "local_gap": local_stats["gap"],

        "pair_positive_mean": pair_stats["positive_mean"],
        "pair_negative_mean": pair_stats["negative_mean"],
        "pair_positive_std": pair_stats["positive_std"],
        "pair_negative_std": pair_stats["negative_std"],
        "pair_gap": pair_stats["gap"],
    }