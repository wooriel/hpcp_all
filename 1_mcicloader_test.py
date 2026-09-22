import os
from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import ci_mcicdataset


parser = argparse.ArgumentParser()
parser.add_argument("--midi_dir", type=str, default="MCIC/MCIC DATA/MIDI FILES", help="data directory name")
parser.add_argument("--csv_dir", type=str, default="MCIC/MCIC DATA", help="data directory name")
parser.add_argument("--csv_name", type=str, default="MCIC.csv")
parser.add_argument("--cache_dir", type=str, default="cache", help="cache directory name")
parser.add_argument("--rep", type=str, default="hpcp", help="representation of the input")
args = parser.parse_args()

# path
base_path = Path.cwd()
midi_path = os.path.join(base_path, "data", args.midi_dir)
pair_csv_path = os.path.join(base_path, "data", args.csv_dir, args.csv_name)
cache_dir = os.path.join(base_path, args.cache_dir, "_".join([args.rep, args.cache_dir]))
cache_path = os.path.join(cache_dir, "MCIC")

# load dataset
mcic_dataset = ci_mcicdataset.MCICDataset(pair_csv_path=pair_csv_path, dpath=midi_path, inp_rep=args.rep.upper(), crop_len=4375, cache_rep=True, cache_dir=cache_path)

mcic_loader = DataLoader(mcic_dataset, batch_size=4, num_workers=4, pin_memory=True) # pin_memory batches into page-locked cpu ram (fast transfer)

for batch in mcic_loader:
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