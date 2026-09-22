import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path


def parse_train_log(log_path):
    rows = []
    current_epoch = None
    pending_val = None

    patterns = {
        "loss": r"(?:Train Loss|Val Loss):\s*(-?\d+(?:\.\d+)?)",
        "pos": r"Pos:\s*(-?\d+(?:\.\d+)?)",
        "neg": r"Neg:\s*(-?\d+(?:\.\d+)?)",
        "gap": r"Gap:\s*(-?\d+(?:\.\d+)?)",
        "auc": r"AUC:\s*(-?\d+(?:\.\d+)?)",
        "mrr": r"MRR:\s*(-?\d+(?:\.\d+)?)",
        "map": r"MAP:\s*(-?\d+(?:\.\d+)?)",
        "top1": r"Top1:\s*(-?\d+(?:\.\d+)?)",
        "topk": r"Topk:\s*(-?\d+(?:\.\d+)?)",
        "triplet_loss": r"Triplet Loss:\s*(-?\d+(?:\.\d+)?)",
        "ap_loss": r"AP Loss:\s*(-?\d+(?:\.\d+)?)",
        "local_loss": r"Local Loss:\s*(-?\d+(?:\.\d+)?)",
    }

    def extract_metrics(line, row):
        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match is not None:
                row[key] = float(match.group(1))

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("Epoch"): # Train line
                match = re.search(r"Epoch\s+(\d+)", line)
                if match is None:
                    continue

                current_epoch = int(match.group(1))
                row = {"epoch": current_epoch, "split": "train"}

                extract_metrics(line, row)
                rows.append(row)
            
            elif line.startswith("Val Loss"): # Validation first line
                temp = {"epoch": current_epoch, "split": "val"}
                extract_metrics(line, temp)

                if "auc" in temp: # new format
                    rows.append(temp)
                    temp = None


            elif line.startswith("AUC:"):
                if temp is None:
                    continue

                extract_metrics(line, temp)

                rows.append(temp)
                temp = None

    return pd.DataFrame(rows)


def plot_metric_trend(df, y_key, save_dir="plots", y_decimals=4):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_df = df[df["split"] == "train"].sort_values("epoch")
    val_df = df[df["split"] == "val"].sort_values("epoch")

    plt.figure(figsize=(6, 6))

    if y_key in train_df.columns:
        plt.plot(
            train_df["epoch"],
            train_df[y_key],
            marker="o",
            markersize=2,
            linestyle="-",
            label="Train",
        )

    if y_key in val_df.columns:
        plt.plot(
            val_df["epoch"],
            val_df[y_key],
            marker="o",
            markersize=2,
            linestyle="-",
            label="Validation",
        )

    plt.xlabel("Epoch")
    plt.ylabel(y_key.upper())
    plt.title(f"{y_key.upper()} Trend")
    plt.legend()
    plt.grid(True, alpha=0.1)

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter(f"%.{y_decimals}f"))

    plt.tight_layout()

    save_path = save_dir / f"{y_key}_trend.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

    print(f"Saved: {save_path}")


def parse_eval_log(log_path):
    records = []
    current_epoch = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Epoch 5, Epoch 005, Epoch [5], etc.
            if line.lower().startswith("epoch"):
                match = re.search(r"(\d+)", line)

                if match is not None:
                    current_epoch = int(match.group(1))

                continue

            if current_epoch is None:
                continue

            parts = [part.strip() for part in line.split("|")]

            if len(parts) < 2:
                continue

            record = {
                "epoch": current_epoch,
                "dataset": parts[0].split()[0],
            }

            for part in parts[1:]:
                if ":" not in part:
                    continue

                key, value = part.split(":", 1)

                key = key.strip().lower().replace(" ", "_")
                value = value.strip()

                try:
                    record[key] = float(value)
                except ValueError:
                    continue

            records.append(record)

    return records


def plot_eval_metric_by_dataset(records, metric, save_dir="eval_plots"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(set(record["dataset"] for record in records))

    for dataset in datasets:
        dataset_records = [record for record in records if record["dataset"] == dataset and metric in record]
        dataset_records = sorted(dataset_records, key=lambda x: x["epoch"])

        if len(dataset_records) == 0:
            continue

        epochs = [record["epoch"] for record in dataset_records]
        values = [record[metric] for record in dataset_records]

        plt.figure(figsize=(6, 6))
        plt.plot(epochs, values, marker="o", markersize=3)

        plt.xlabel("Epoch")
        plt.ylabel(metric.upper())
        plt.title(f"{dataset} - {metric.upper()}")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()

        save_path = save_dir / f"{dataset}_{metric}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()