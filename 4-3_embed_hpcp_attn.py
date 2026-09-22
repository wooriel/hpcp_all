import os
import argparse
from pathlib import Path
import torch
from dataset import cs_shsdataset, cs_cover80dataset, ci_copyright40dataset, ci_mcicdataset, pndatacosdataset
from torch.utils.data import DataLoader
from models import a_vittokenizer, b_selfattention, c_self_sim_model
from embedding import evaluate_embedding_hpcp_vit
from checkpoint import load_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument("--csv_pref", type=str, default="pair")
parser.add_argument("--exp_name", type=str, default="hpcp_grad_attn_v1", help="name of training experiment")
parser.add_argument("--cache_dir", type=str, default="cache", help="cache directory name")
parser.add_argument("--rep", type=str, default="hpcp", help="representation of the input")
args = parser.parse_args()

# data path
base_path = Path.cwd()
shs_data_path = os.path.join(base_path, "data", "shs-100k")
shs_pair_csv_path = os.path.join(shs_data_path, "pair_test", "_".join([args.csv_pref, "test.csv"]))
cover80_data_path = os.path.join(base_path, "data", "covers80_32k")
copyright40_data_path = os.path.join(base_path, "data", "copyright40", "CopyrightCases")
mcic_midi_path = os.path.join(base_path, "data", "MCIC", "MCIC DATA", "MIDI FILES")
datacos_data_path = os.path.join(base_path, "data", "da_tacos")

# csv path
copyright40_pair_csv_path = os.path.join(copyright40_data_path, "copyright40.csv")
mcic_pair_csv_path = os.path.join(base_path, "data", "MCIC", "MCIC DATA", "MCIC.csv")
datacos_csv_path = os.path.join(base_path, "data", "da_tacos", "da-tacos_metadata", "{}.csv".format("tri_test"))

cache_dir = os.path.join(base_path, args.cache_dir, "_".join([args.rep, args.cache_dir]))
cache_test_path = os.path.join(cache_dir, "shs100_test")
c80_cache_path = os.path.join(cache_dir, "covers80")
cp40_cache_path = os.path.join(cache_dir, "copyright40_Full")
mcic_cache_path = os.path.join(cache_dir, "MCIC")
datacos_cache_path = os.path.join(cache_dir, "da_tacos")

checkpoint_dir = Path("checkpoints") / args.exp_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)
log_path = checkpoint_dir / "log_eval.txt"

# load dataset
shs_test_dataset = cs_shsdataset.SHSDataset(pair_csv_path=shs_pair_csv_path, dpath=shs_data_path, inp_rep=args.rep.upper(), crop_len=4375, split="test", cache_rep=True, cache_dir=cache_test_path)
cover80_dataset = cs_cover80dataset.Cover80Dataset(lpath_src="list1.list", lpath_tgt="list2.list", dpath=cover80_data_path, inp_rep=args.rep.upper(), crop_len=4375, seed=2026, cache_rep=True, cache_dir=c80_cache_path)
copyright40_full_dataset = ci_copyright40dataset.Copyright40Dataset(pair_csv_path=copyright40_pair_csv_path, dpath=copyright40_data_path, inp_rep=args.rep.upper(), division="Full", crop_len=4375, cache_rep=True, cache_dir=cp40_cache_path)
mcic_dataset = ci_mcicdataset.MCICDataset(pair_csv_path=mcic_pair_csv_path, dpath=mcic_midi_path, inp_rep=args.rep.upper(), crop_len=4375, cache_rep=True, cache_dir=mcic_cache_path)
datacos_dataset = pndatacosdataset.PNDATACOSDataset(csv_path=datacos_csv_path, dpath=datacos_data_path, inp_rep=args.rep.upper(), crop_len=4375, split="test", seed=2026, cache_dir=datacos_cache_path)

shs_test_loader = DataLoader(shs_test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
cover80_loader = DataLoader(cover80_dataset, batch_size=16, num_workers=4, shuffle=False, pin_memory=True)
copyright40_full_loader = DataLoader(copyright40_full_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
mcic_loader = DataLoader(mcic_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
datacos_loader = DataLoader(datacos_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
tokenizer = a_vittokenizer.RawTokenizer()
selfattn = b_selfattention.SelfAttentionEncoder(nhead=1, d_model=64, num_layers=2, dim_feedforward=256, dropout=0.1)
model = c_self_sim_model.HPCPSelfAttentionModel(tokenizer=tokenizer, self_attention=selfattn, nhead=1, d_model=64, num_pool=4).to(device=device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4,
)

best_loss = float("inf")

checkpoint_path = checkpoint_dir / "last.pt"
# checkpoint_path = checkpoint_dir / "best.pt"


if checkpoint_path.is_file():
    start_epoch, loaded_loss = load_checkpoint(
        model=model,
        optimizer=optimizer,
        path=checkpoint_path,
        device=device,
    )
    if loaded_loss is not None:
        best_loss = loaded_loss
else:
    print("No checkpoint found. Starting from epoch 1.")

# Evaluation on shs-100K test dataset pair
shs_test_metrics = evaluate_embedding_hpcp_vit(
    model=model,
    dataloader=shs_test_loader,
    device=device,
)

test_dataset = "shs_test_pairs"

print(
    f"Tokenizer Pos Cos: {shs_test_metrics['tokenizer_positive_mean']:.4f} | "
    f"Neg Cos: {shs_test_metrics['tokenizer_negative_mean']:.4f} | "
    f"Pos Std: {shs_test_metrics['tokenizer_positive_std']:.4f} | "
    f"Neg Std: {shs_test_metrics['tokenizer_negative_std']:.4f} | "
    f"Gap: {shs_test_metrics['tokenizer_gap']:.4f}\n"

    f"Global Pos Cos: {shs_test_metrics['global_positive_mean']:.4f} | "
    f"Neg Cos: {shs_test_metrics['global_negative_mean']:.4f} | "
    f"Pos Std: {shs_test_metrics['global_positive_std']:.4f} | "
    f"Neg Std: {shs_test_metrics['global_negative_std']:.4f} | "
    f"Gap: {shs_test_metrics['global_gap']:.4f}\n"

    f"Local Pos Sim: {shs_test_metrics['local_positive_mean']:.4f} | "
    f"Neg Sim: {shs_test_metrics['local_negative_mean']:.4f} | "
    f"Pos Std: {shs_test_metrics['local_positive_std']:.4f} | "
    f"Neg Std: {shs_test_metrics['local_negative_std']:.4f} | "
    f"Gap: {shs_test_metrics['local_gap']:.4f}\n"

    f"Pair Pos Sim: {shs_test_metrics['pair_positive_mean']:.4f} | "
    f"Neg Sim: {shs_test_metrics['pair_negative_mean']:.4f} | "
    f"Pos Std: {shs_test_metrics['pair_positive_std']:.4f} | "
    f"Neg Std: {shs_test_metrics['pair_negative_std']:.4f} | "
    f"Gap: {shs_test_metrics['pair_gap']:.4f}"
)

# Evaluation on cover80 dataset pair
c80_test_metrics = evaluate_embedding_hpcp_vit(
    model=model,
    dataloader=cover80_loader,
    device=device,
)

test_dataset = "cover80"

print(
    f"Tokenizer Pos Cos: {c80_test_metrics['tokenizer_positive_mean']:.4f} | "
    f"Neg Cos: {c80_test_metrics['tokenizer_negative_mean']:.4f} | "
    f"Pos Std: {c80_test_metrics['tokenizer_positive_std']:.4f} | "
    f"Neg Std: {c80_test_metrics['tokenizer_negative_std']:.4f} | "
    f"Gap: {c80_test_metrics['tokenizer_gap']:.4f}\n"

    f"Global Pos Cos: {c80_test_metrics['global_positive_mean']:.4f} | "
    f"Neg Cos: {c80_test_metrics['global_negative_mean']:.4f} | "
    f"Pos Std: {c80_test_metrics['global_positive_std']:.4f} | "
    f"Neg Std: {c80_test_metrics['global_negative_std']:.4f} | "
    f"Gap: {c80_test_metrics['global_gap']:.4f}\n"

    f"Local Pos Sim: {c80_test_metrics['local_positive_mean']:.4f} | "
    f"Neg Sim: {c80_test_metrics['local_negative_mean']:.4f} | "
    f"Pos Std: {c80_test_metrics['local_positive_std']:.4f} | "
    f"Neg Std: {c80_test_metrics['local_negative_std']:.4f} | "
    f"Gap: {c80_test_metrics['local_gap']:.4f}\n"

    f"Pair Pos Sim: {c80_test_metrics['pair_positive_mean']:.4f} | "
    f"Neg Sim: {c80_test_metrics['pair_negative_mean']:.4f} | "
    f"Pos Std: {c80_test_metrics['pair_positive_std']:.4f} | "
    f"Neg Std: {c80_test_metrics['pair_negative_std']:.4f} | "
    f"Gap: {c80_test_metrics['pair_gap']:.4f}"
)

# Evaluation on datacos dataset pair
datacos_test_metrics = evaluate_embedding_hpcp_vit(
    model=model,
    dataloader=datacos_loader,
    device=device,
)

test_dataset = "datacos"

print(
    f"Tokenizer Pos Cos: {datacos_test_metrics['tokenizer_positive_mean']:.4f} | "
    f"Neg Cos: {datacos_test_metrics['tokenizer_negative_mean']:.4f} | "
    f"Pos Std: {datacos_test_metrics['tokenizer_positive_std']:.4f} | "
    f"Neg Std: {datacos_test_metrics['tokenizer_negative_std']:.4f} | "
    f"Gap: {datacos_test_metrics['tokenizer_gap']:.4f}\n"

    f"Global Pos Cos: {datacos_test_metrics['global_positive_mean']:.4f} | "
    f"Neg Cos: {datacos_test_metrics['global_negative_mean']:.4f} | "
    f"Pos Std: {datacos_test_metrics['global_positive_std']:.4f} | "
    f"Neg Std: {datacos_test_metrics['global_negative_std']:.4f} | "
    f"Gap: {datacos_test_metrics['global_gap']:.4f}\n"

    f"Local Pos Sim: {datacos_test_metrics['local_positive_mean']:.4f} | "
    f"Neg Sim: {datacos_test_metrics['local_negative_mean']:.4f} | "
    f"Pos Std: {datacos_test_metrics['local_positive_std']:.4f} | "
    f"Neg Std: {datacos_test_metrics['local_negative_std']:.4f} | "
    f"Gap: {datacos_test_metrics['local_gap']:.4f}\n"

    f"Pair Pos Sim: {datacos_test_metrics['pair_positive_mean']:.4f} | "
    f"Neg Sim: {datacos_test_metrics['pair_negative_mean']:.4f} | "
    f"Pos Std: {datacos_test_metrics['pair_positive_std']:.4f} | "
    f"Neg Std: {datacos_test_metrics['pair_negative_std']:.4f} | "
    f"Gap: {datacos_test_metrics['pair_gap']:.4f}"
)