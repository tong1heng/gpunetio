#!/usr/bin/env python3
import re
import csv
import argparse
from pathlib import Path


def parse_threads_from_filename(path: Path):
    """
    从文件名解析 threads 数量：
    例如：
      1b1t.log  -> 1
      1b2t.log  -> 2
      1b16t.log -> 16
    """
    m = re.search(r"(\d+)b(\d+)t", path.name)
    if m:
        return int(m.group(2))

    return None


def parse_scope_from_text(text: str):
    """
    从日志中解析 scope：
    例如：
      THREAD scope
      CUDA_BLOCK scope
    """
    m = re.search(r"\b([A-Z_]+)\s+scope\b", text)
    if m:
        return m.group(1)

    return ""


def parse_log_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")

    threads = parse_threads_from_filename(path)
    scope = parse_scope_from_text(text)

    rows = []

    # 正常完整结果行，例如：
    # 1           2048              0.000942            0.117751            17.392639
    normal_line_pattern = re.compile(
        r"^\s*"
        r"(?P<bytes>\d+)\s+"
        r"(?P<iterations>\d+)\s+"
        r"(?P<bw>[0-9]*\.?[0-9]+)\s+"
        r"(?P<msg>[0-9]*\.?[0-9]+)\s+"
        r"(?P<cuda>[0-9]*\.?[0-9]+)"
        r"\s*$"
    )

    for line in text.splitlines():
        m = normal_line_pattern.match(line)
        if not m:
            continue

        rows.append({
            "threads": threads,
            "bytes": int(m.group("bytes")),
            "iterations": int(m.group("iterations")),
            "bw_average_Gbps": float(m.group("bw")),
            "msg_rate_Mpps": float(m.group("msg")),
            "cuda_kernel_ms": float(m.group("cuda")),
            "scope": scope,
            "file": path.name,
        })

    # 处理被日志插入打断的结果行，例如：
    #
    # 262144      2048              109.867557            0.052389          Tue May ...
    #   39.092224
    broken_line_pattern = re.compile(
        r"^\s*"
        r"(?P<bytes>\d+)\s+"
        r"(?P<iterations>\d+)\s+"
        r"(?P<bw>[0-9]*\.?[0-9]+)\s+"
        r"(?P<msg>[0-9]*\.?[0-9]+)"
        r"[^\n]*\n"
        r"\s*(?P<cuda>[0-9]*\.?[0-9]+)\s*$",
        re.MULTILINE
    )

    existing = {
        (
            r["bytes"],
            r["iterations"],
            r["bw_average_Gbps"],
            r["msg_rate_Mpps"],
        )
        for r in rows
    }

    for m in broken_line_pattern.finditer(text):
        key = (
            int(m.group("bytes")),
            int(m.group("iterations")),
            float(m.group("bw")),
            float(m.group("msg")),
        )

        if key in existing:
            continue

        rows.append({
            "threads": threads,
            "bytes": int(m.group("bytes")),
            "iterations": int(m.group("iterations")),
            "bw_average_Gbps": float(m.group("bw")),
            "msg_rate_Mpps": float(m.group("msg")),
            "cuda_kernel_ms": float(m.group("cuda")),
            "scope": scope,
            "file": path.name,
        })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Parse all GPUNetIO put_bw logs in a directory into one CSV"
    )

    parser.add_argument(
        "-d",
        "--dir",
        default="/home/tyh/gpunetio/test/191c_191s/put_bw/thread/",
        help="log directory"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="put_bw_thread_summary.csv",
        help="output csv file"
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="scan log files recursively"
    )

    args = parser.parse_args()

    log_dir = Path(args.dir)

    if not log_dir.exists():
        raise FileNotFoundError(f"Directory not found: {log_dir}")

    if args.recursive:
        log_files = sorted(log_dir.rglob("*.log"))
    else:
        log_files = sorted(log_dir.glob("*.log"))

    all_rows = []

    for log_file in log_files:
        rows = parse_log_file(log_file)
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (r["threads"] if r["threads"] is not None else -1, r["bytes"]))

    fieldnames = [
        "threads",
        "bytes",
        "iterations",
        "bw_average_Gbps",
        "msg_rate_Mpps",
        "cuda_kernel_ms",
        "scope",
        "file",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Parsed {len(log_files)} log files")
    print(f"Generated {len(all_rows)} rows")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()