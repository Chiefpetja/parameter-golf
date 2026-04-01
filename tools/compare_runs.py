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
        f"seq={row.get('train_seq_len', '')} | "
        f"tokens={row.get('train_batch_tokens', '')} | "
        f"subm_zlib={row.get('submission_int8_zlib_bytes', '')}"
    )


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    rows = load_rows(path)

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