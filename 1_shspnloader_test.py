import os
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import cs_shsdataset


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="shs-100k", help="data directory name")
parser.add_argument("--csv_pref", type=str, default="pair")
parser.add_argument("--cache_dir", type=str, default="cache", help="cache directory name")
parser.add_argument("--rep", type=str, default="hpcp", help="representation of the input")
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

# load dataset
shs_train_dataset = cs_shsdataset.PNSHSDataset(pair_csv_path=pair_csv_train_path, dpath=data_path, inp_rep=args.rep.upper(), crop_len=4375, split="train", cache_rep=True, cache_dir=cache_train_path)
shs_val_dataset = cs_shsdataset.PNSHSDataset(pair_csv_path=pair_csv_valid_path, dpath=data_path, inp_rep=args.rep.upper(), crop_len=4375, split="validate", cache_rep=True, cache_dir=cache_validate_path)
shs_test_dataset = cs_shsdataset.PNSHSDataset(pair_csv_path=pair_csv_test_path, dpath=data_path, inp_rep=args.rep.upper(), crop_len=4375, split="test", cache_rep=True, cache_dir=cache_test_path)

shs_train_loader = DataLoader(shs_train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
shs_val_loader = DataLoader(shs_val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True) # pin_memory batches into page-locked cpu ram (fast transfer)
shs_test_loader = DataLoader(shs_test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

for batch in tqdm(shs_train_loader):
    src_work_id = batch["src_work_id"]
    pos_work_id = batch["pos_work_id"]
    neg_work_id = batch["neg_work_id"]
    src_label = batch["src_label"]
    tgt_label = batch["pos_label"]
    neg_label = batch["neg_label"]
    orig_inp = batch["orig_inp"]
    pos_inp = batch["pos_inp"]
    neg_inp = batch["neg_inp"]

    print(src_work_id.shape)
    print(src_label.shape)
    print(orig_inp.shape)


for batch in tqdm(shs_val_loader):
    src_work_id = batch["src_work_id"]
    pos_work_id = batch["pos_work_id"]
    neg_work_id = batch["neg_work_id"]
    src_label = batch["src_label"]
    tgt_label = batch["pos_label"]
    neg_label = batch["neg_label"]
    orig_inp = batch["orig_inp"]
    pos_inp = batch["pos_inp"]
    neg_inp = batch["neg_inp"]

    print(src_work_id.shape)
    print(src_label.shape)
    print(orig_inp.shape)


for batch in tqdm(shs_test_loader):
    src_work_id = batch["src_work_id"]
    pos_work_id = batch["pos_work_id"]
    neg_work_id = batch["neg_work_id"]
    src_label = batch["src_label"]
    tgt_label = batch["pos_label"]
    neg_label = batch["neg_label"]
    orig_inp = batch["orig_inp"]
    pos_inp = batch["pos_inp"]
    neg_inp = batch["neg_inp"]

    print(src_work_id.shape)
    print(src_label.shape)
    print(orig_inp.shape)