import os
from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import pndatacosdataset


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="da_tacos", help="data directory name")
parser.add_argument("--hpcp_dir", type=str, default="da-tacos_benchmark_subset_hpcp", help="hpcp directory name")
parser.add_argument("--csv_dir", type=str, default="da-tacos_metadata", help="directory of metadata")
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--cache_dir", type=str, default="hpcp_cache", help="cache directory name")
parser.add_argument("--rep", type=str, default="hpcp", help="representation of the input")
args = parser.parse_args()

# path
base_path = Path.cwd()
data_path = os.path.join(base_path, "data", args.data_dir, args.hpcp_dir)
csv_path = os.path.join(base_path, "data", args.data_dir, args.csv_dir, "{}.csv".format(args.split))
cache_path = os.path.join(base_path, "cache", args.cache_dir, args.data_dir)

# load dataset
datacos_dataset = pndatacosdataset.PNDATACOSDataset(csv_path=csv_path, dpath=data_path, inp_rep=args.rep.upper(), crop_len=4375, split="train", cache_rep=True, cache_dir=cache_path)

datacos_loader = DataLoader(datacos_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True) # pin_memory batches into page-locked cpu ram (fast transfer)

print(len(datacos_dataset))
# for batch in datacos_loader:
#     work = batch["orig_inp"]
#     perf = batch["cover_inp"]
#     decision = batch["decision"]

#     print(work.shape)
#     print(perf.shape)
#     print(decision.shape)