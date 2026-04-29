import csv
import os
import sys

CSV_PATH = "run_history.csv"


def to_float(value, default=float("inf")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=10**18):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default

def to_str(value, default=""):
    if value is None:
        return default
    return str(value)


def compute_long_run_factor(row, baseline_row):
    try:
        row_gain = float(row.get("bpb_gain_per_step", ""))
        row_steps = float(row.get("step_reached", ""))
        base_gain = float(baseline_row.get("bpb_gain_per_step", ""))
        base_steps = float(baseline_row.get("step_reached", ""))

        if base_gain <= 0 or base_steps <= 0:
            return ""

        step_gain_factor = row_gain / base_gain
        step_count_factor = row_steps / base_steps
        long_run_factor = step_gain_factor * step_count_factor
        return f"{long_run_factor:.6f}"
    except (TypeError, ValueError, ZeroDivisionError):
        return ""

def load_rows(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"No rows found in: {path}")
        sys.exit(1)

    return rows


def fmt_run(row):
    return (
        f"run={row.get('run_id', '')} | "
        f"val_bpb={row.get('val_bpb', '')} | "
        f"step={row.get('step_reached', '')} | "
        f"avg_ms={row.get('step_avg_ms', '')} | "
        f"bpb_gain_step={row.get('bpb_gain_per_step', '')} | "
        f"bpb_gain_sec={row.get('bpb_gain_per_sec', '')} | "
        f"long_run={row.get('long_run_factor', '')} | "
        f"seq={row.get('train_seq_len', '')} | "
        f"tokens={row.get('train_batch_tokens', '')} | "
        f"subm_zlib={row.get('submission_int8_zlib_bytes', '')} | "
        f"notes={row.get('notes', '')}"
    )


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    rows = load_rows(path)

    baseline_candidates = [
        r for r in rows
        if "baseline" in to_str(r.get("notes", "")).lower()
        and r.get("bpb_gain_per_step")
        and r.get("step_reached")
    ]

    baseline_row = None
    if baseline_candidates:
        # Take the best baseline by val_bpb
        baseline_row = min(baseline_candidates, key=lambda r: to_float(r.get("val_bpb")))

    for r in rows:
        r["long_run_factor"] = compute_long_run_factor(r, baseline_row) if baseline_row else ""
    valid_bpb = [r for r in rows if r.get("val_bpb")]
    valid_speed = [r for r in rows if r.get("step_avg_ms")]
    valid_size = [r for r in rows if r.get("submission_int8_zlib_bytes")]

    print(f"Loaded runs: {len(rows)}")
    print("-" * 80)

    if valid_bpb:
        best_bpb = min(valid_bpb, key=lambda r: to_float(r.get("val_bpb")))
        print("Best val_bpb:")
        print(" ", fmt_run(best_bpb))
        print()

    if valid_speed:
        fastest = min(valid_speed, key=lambda r: to_float(r.get("step_avg_ms")))
        print("Fastest step_avg_ms:")
        print(" ", fmt_run(fastest))
        print()

    if valid_size:
        smallest = min(valid_size, key=lambda r: to_int(r.get("submission_int8_zlib_bytes")))
        print("Smallest int8+zlib submission:")
        print(" ", fmt_run(smallest))
        print()

    valid_long_run = [r for r in rows if r.get("long_run_factor")]
    if valid_long_run:
        best_long_run = max(valid_long_run, key=lambda r: to_float(r.get("long_run_factor"), default=float("-inf")))
        print("Best long_run_factor:")
        print(" ", fmt_run(best_long_run))
        print()
        
    if valid_bpb:
        print("Top 5 by val_bpb:")
        ranked = sorted(valid_bpb, key=lambda r: to_float(r.get("val_bpb")))
        for i, row in enumerate(ranked[:5], start=1):
            print(f"  {i}. {fmt_run(row)}")
        print()

    print("Latest run:")
    print(" ", fmt_run(rows[-1]))


if __name__ == "__main__":
    main()