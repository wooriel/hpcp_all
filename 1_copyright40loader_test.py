import os
from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import ci_copyright40dataset


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="copyright40/CopyrightCases", help="data directory name")
parser.add_argument("--csv_name", type=str, default="copyright40.csv")
parser.add_argument("--division", type=str, default="Full", help="one of full(full song) and melody (only melody)")
parser.add_argument("--cache_dir", type=str, default="cache", help="cache directory name")
parser.add_argument("--rep", type=str, default="hpcp", help="representation of the input")
args = parser.parse_args()

# path
base_path = Path.cwd()
data_path = os.path.join(base_path, "data", args.data_dir)
csv_path = os.path.join(data_path, args.csv_name)
cache_dir = os.path.join(base_path, args.cache_dir, "_".join([args.rep, args.cache_dir]))
cache_path_full = os.path.join(cache_dir, "_".join(["copyright40", args.division]))
cach_path_melody = os.path.join(cache_dir, "_".join(["copyright40", "Melody"]))

# load dataset
copyright40_full_dataset = ci_copyright40dataset.Copyright40Dataset(pair_csv_path=csv_path, dpath=data_path, inp_rep=args.rep.upper(), division=args.division, crop_len=4375, cache_rep=True, cache_dir=cache_path_full)
copyright40_melody_dataset = ci_copyright40dataset.Copyright40Dataset(pair_csv_path=csv_path, dpath=data_path, inp_rep=args.rep.upper(), division="Melody", crop_len=4375, cache_rep=True, cache_dir=cach_path_melody)

copy40_full_loader = DataLoader(copyright40_full_dataset, batch_size=16, num_workers=4, pin_memory=True)
copyright40_melody_loader = DataLoader(copyright40_melody_dataset, batch_size=16, num_workers=4, pin_memory=True)

for batch in copy40_full_loader:
    orig_input = batch["orig_inp"]
    cover_input = batch["cover_inp"]
    decision = batch["decision"]
    work_id = batch["work_id"]
    src_perf_id = batch["src_perf_id"]
    tgt_perf_id = batch["tgt_perf_id"]

    print(orig_input.shape)
    print(cover_input.shape)
    print(decision.shape)
    print(src_perf_id)
    print(tgt_perf_id)


for batch in copyright40_melody_loader:
    orig_input = batch["orig_inp"]
    cover_input = batch["cover_inp"]
    decision = batch["decision"]
    work_id = batch["work_id"]
    src_perf_id = batch["src_perf_id"]
    tgt_perf_id = batch["tgt_perf_id"]

    print(orig_input.shape)
    print(cover_input.shape)
    print(decision.shape)
    print(work_id)
    print(src_perf_id)
    print(tgt_perf_id)