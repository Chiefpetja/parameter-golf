import csv
import os
import re
import sys
from datetime import datetime

PATTERNS = {
    "run_id": re.compile(r"logs[\\/](.+?)\.txt"),
    "train_cfg": re.compile(
        r"train_batch_tokens:(\d+)\s+train_seq_len:(\d+)\s+iterations:(\d+)\s+warmup_steps:(\d+)\s+max_wallclock_seconds:([0-9.]+)"
    ),
    "attention": re.compile(r"attention_mode:(\w+)\s+num_heads:(\d+)\s+num_kv_heads:(\d+)"),
    "sdp": re.compile(r"sdp_backends:cudnn=(\w+)\s+flash=(\w+)\s+mem_efficient=(\w+)\s+math=(\w+)"),
    "val_step": re.compile(r"step:(\d+)/(\d+)\s+val_loss:([0-9.]+)\s+val_bpb:([0-9.]+)"),
    "train_step": re.compile(r"step:(\d+)/(\d+)\s+train_loss:([0-9.]+).*?step_avg:([0-9.]+)ms"),
    "stop": re.compile(r"stopping_early:.*?step:(\d+)/(\d+)"),
    "peak_mem": re.compile(r"peak memory allocated:\s+(\d+)\s+MiB reserved:\s+(\d+)\s+MiB"),
    "submission_size": re.compile(r"Total submission size:\s+(\d+)\s+bytes"),
    "submission_int8": re.compile(r"Total submission size int8\+zlib:\s+(\d+)\s+bytes"),
}

FIELDNAMES = [
    "timestamp",
    "run_id",
    "train_seq_len",
    "train_batch_tokens",
    "iterations",
    "max_wallclock_seconds",
    "attention_mode",
    "num_heads",
    "num_kv_heads",
    "flash_sdp",
    "mem_efficient_sdp",
    "math_sdp",
    "step_reached",
    "val_loss",
    "val_bpb",
    "train_loss_last",
    "step_avg_ms",
    "peak_mem_alloc_mib",
    "peak_mem_reserved_mib",
    "submission_size_bytes",
    "submission_int8_zlib_bytes",
    "start_val_bpb",
    "bpb_gain_pct",
    "bpb_gain_per_step",
    "bpb_gain_per_sec",
    "train_loss_gain_per_step",
    "notes",
]

def parse_log(text: str):
    row = {k: "" for k in FIELDNAMES}
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")

    m = PATTERNS["run_id"].search(text)
    if m:
        row["run_id"] = m.group(1)

    m = PATTERNS["train_cfg"].search(text)
    if m:
        row["train_batch_tokens"] = m.group(1)
        row["train_seq_len"] = m.group(2)
        row["iterations"] = m.group(3)
        row["max_wallclock_seconds"] = m.group(5)

    m = PATTERNS["attention"].search(text)
    if m:
        row["attention_mode"] = m.group(1)
        row["num_heads"] = m.group(2)
        row["num_kv_heads"] = m.group(3)

    m = PATTERNS["sdp"].search(text)
    if m:
        row["flash_sdp"] = m.group(2)
        row["mem_efficient_sdp"] = m.group(3)
        row["math_sdp"] = m.group(4)

    train_matches = list(PATTERNS["train_step"].finditer(text))
    if train_matches:
        last_train = train_matches[-1]
        row["step_reached"] = last_train.group(1)
        row["train_loss_last"] = last_train.group(3)
        row["step_avg_ms"] = last_train.group(4)

    val_matches = list(PATTERNS["val_step"].finditer(text))
    if val_matches:
        first_val = val_matches[0]
        last_val = val_matches[-1]
        row["start_val_bpb"] = first_val.group(4)
        row["step_reached"] = last_val.group(1)
        row["val_loss"] = last_val.group(3)
        row["val_bpb"] = last_val.group(4)

    m = PATTERNS["stop"].search(text)
    if m:
        row["step_reached"] = m.group(1)

    m = PATTERNS["peak_mem"].search(text)
    if m:
        row["peak_mem_alloc_mib"] = m.group(1)
        row["peak_mem_reserved_mib"] = m.group(2)

    m = PATTERNS["submission_size"].search(text)
    if m:
        row["submission_size_bytes"] = m.group(1)

    m = PATTERNS["submission_int8"].search(text)
    if m:
        row["submission_int8_zlib_bytes"] = m.group(1)

    # Derived metrics
    try:
        start_val_bpb = float(row["start_val_bpb"])
        end_val_bpb = float(row["val_bpb"])
        step_reached = float(row["step_reached"])
        max_wallclock_seconds = float(row["max_wallclock_seconds"])

        bpb_gain = start_val_bpb - end_val_bpb
        row["bpb_gain_pct"] = f"{(bpb_gain / start_val_bpb):.6f}" if start_val_bpb != 0 else ""
        row["bpb_gain_per_step"] = f"{(bpb_gain / step_reached):.8f}" if step_reached > 0 else ""
        row["bpb_gain_per_sec"] = f"{(bpb_gain / max_wallclock_seconds):.8f}" if max_wallclock_seconds > 0 else ""
    except (ValueError, TypeError, KeyError):
        pass

    try:
        if train_matches:
            first_train_loss = float(train_matches[0].group(3))
            last_train_loss = float(row["train_loss_last"])
            step_reached = float(row["step_reached"])
            train_loss_gain = first_train_loss - last_train_loss
            row["train_loss_gain_per_step"] = f"{(train_loss_gain / step_reached):.8f}" if step_reached > 0 else ""
    except (ValueError, TypeError, KeyError):
        pass

    return row

def append_csv(path: str, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def print_summary(csv_path: str):
    rows = []
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    valid = [r for r in rows if r.get("val_bpb")]
    if valid:
        best = min(valid, key=lambda r: float(r["val_bpb"]))
        print(f"Best val_bpb so far: {best['val_bpb']} (run {best['run_id']}, step {best['step_reached']})")

    latest = rows[-1]
    print(f"Latest run: {latest['run_id']} | val_bpb={latest['val_bpb']} | step={latest['step_reached']} | avg_ms={latest['step_avg_ms']}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/log_run.py <path_to_log> [notes]")
        sys.exit(1)

    log_path = sys.argv[1]
    notes = sys.argv[2] if len(sys.argv) > 2 else ""

    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    row = parse_log(text)
    row["notes"] = notes

    csv_path = "run_history.csv"
    append_csv(csv_path, row)
    print(f"Logged run to {csv_path}")
    print_summary(csv_path)

if __name__ == "__main__":
    main()