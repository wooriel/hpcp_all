import os
import argparse
from pathlib import Path
import torch
from dataset import cs_shsdataset
from torch.utils.data import DataLoader
from models import a_vittokenizer, c_self_sim_model, not_used_b_selfattention
from train_tri import train_vit_triplet_one_epoch, validate_vit_triplet_one_epoch
from train_align import log_train_val
from checkpoint import save_checkpoint, load_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="shs-100k", help="data directory name")
parser.add_argument("--csv_pref", type=str, default="pair")
parser.add_argument("--exp_name", type=str, default="cqt_attn_v1", help="name of training experiment")
parser.add_argument("--cache_dir", type=str, default="cache", help="cache directory name")
parser.add_argument("--rep", type=str, default="cqt", help="representation of the input")
args = parser.parse_args()

# path
base_path = Path.cwd()
data_path = os.path.join(base_path, "data", args.data_dir)
pair_csv_train_path = os.path.join(data_path, "pair_train", "_".join([args.csv_pref, "train.csv"]))
pair_csv_valid_path = os.path.join(data_path, "pair_validate", "_".join([args.csv_pref, "validate.csv"]))
pair_csv_test_path = os.path.join(data_path, "pair_test", "_".join([args.csv_pref, "test.csv"]))
cache_dir = os.path.join(base_path, args.cache_dir, "_".join([args.rep, args.cache_dir]))
cache_train_path = os.path.join(cache_dir, "shs100_train")
cache_validate_path = os.path.join(cache_dir, "shs100_validate")
cache_test_path = os.path.join(cache_dir, "shs100_test")

checkpoint_dir = Path("checkpoints") / args.exp_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)
log_path = checkpoint_dir / "log_train.txt"

# load dataset
shs_pn_dataset = cs_shsdataset.SHSDataset(pair_csv_path=pair_csv_train_path, dpath=data_path, inp_rep=args.rep.upper(), crop_len=4375, split="train", cache_rep=True, cache_dir=cache_train_path)
shs_val_dataset = cs_shsdataset.SHSDataset(pair_csv_path=pair_csv_valid_path, dpath=data_path, inp_rep=args.rep.upper(), crop_len=4375, split="validate", cache_rep=True, cache_dir=cache_validate_path)
shs_test_dataset = cs_shsdataset.SHSDataset(pair_csv_path=pair_csv_test_path, dpath=data_path, inp_rep=args.rep.upper(), crop_len=4375, split="test", cache_rep=True, cache_dir=cache_test_path)

print(len(shs_pn_dataset))

shs_train_loader = DataLoader(shs_pn_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
shs_val_loader = DataLoader(shs_val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True) # pin_memory batches into page-locked cpu ram (fast transfer)
shs_test_loader = DataLoader(shs_test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model: CQT linear tokenizer with global and local embedding
tokenizer = a_vittokenizer.CQTLinearTokenizer(d_model=64, patch_freq=24, patch_time=24)
selfattn = not_used_b_selfattention.SelfAttentionEncoder(d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.0)
model = c_self_sim_model.VitSelfAttentionModel(tokenizer=tokenizer, self_attention=selfattn, nfreq=4, d_model=64).to(device=device)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4,
)

# training
num_epochs = 200
validate_every = 1
checkpoint_every = 5

start_epoch = 1
best_val_auc = float("inf")

# checkpoint_path = checkpoint_dir / "best.pt"
checkpoint_path = checkpoint_dir / "last.pt"


if checkpoint_path.is_file():
    start_epoch, loaded_val_loss = load_checkpoint(
        model=model,
        optimizer=optimizer,
        path=checkpoint_path,
        device=device,
    )
    if loaded_val_loss is not None:
        best_val_auc = loaded_val_loss
else:
    print("No checkpoint found. Starting from epoch 1.")


for epoch in range(start_epoch, num_epochs + 1):

    train_metrics = train_vit_triplet_one_epoch(
        model=model,
        dataloader=shs_train_loader,
        optimizer=optimizer,
        device=device,
        glo_margin=0.2,
        loc_margin=0.2,
    )

    print(
        f"Epoch [{epoch}/{num_epochs}] | "
        f"Loss: {train_metrics['loss']:.4f} | "
        f"Pos Cos: {train_metrics['positive_cosine_mean']:.4f} | "
        f"Neg Cos: {train_metrics['negative_cosine_mean']:.4f} | "
        f"Gap: {train_metrics['cosine_gap']:.4f} | "
        f"tri_loss: {train_metrics['tri_loss']:.4f} | ",
        f"rank_loss: {train_metrics['rank_loss']:.4f} | ",
        f"local_loss: {train_metrics['local_loss']:.4f}"
    )

    # Validation
    if epoch % validate_every == 0:

        val_metrics = validate_vit_triplet_one_epoch(
            model=model,
            dataloader=shs_val_loader,
            device=device,
            glo_margin=0.2,
            loc_margin=0.2,
        )

        if val_metrics["roc_auc"] < best_val_auc:
            best_val_auc = val_metrics["roc_auc"]

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=checkpoint_dir / "best.pt",
                val_loss=best_val_auc,
            )

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"Pos: {val_metrics['positive_cosine_mean']:.4f} | "
            f"Neg: {val_metrics['negative_cosine_mean']:.4f} | "
            f"Gap: {val_metrics['cosine_gap']:.4f} | "
            f"AUC {val_metrics['roc_auc']:.4f} | "
            f"MRR: {val_metrics['mrr']:.4f} | ",
            f"MAP: {val_metrics['map']:.4f} | ",
            f"Top1: {val_metrics['top1']:.4f} | ",
            f"Topk: {val_metrics['topk']:.4f} | ",
            f"triplet_acc: {val_metrics['triplet_acc']:.4f} | ",
            f"margin_acc: {val_metrics['margin_acc']:.4f} | ",
            f"tri_loss: {val_metrics['tri_loss']:.4f} | ",
            f"rank_loss: {val_metrics['rank_loss']:.4f} | ",
            f"local_loss: {val_metrics['local_loss']:.4f}"
        )

        log_train_val(log_path=log_path, epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics)

    # -------------------------
    # Checkpoint
    # -------------------------
    if epoch % checkpoint_every == 0:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            path=checkpoint_dir / "model_epoch_{:03d}.pt".format(epoch),
        )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        path=checkpoint_dir / "last.pt",
    )