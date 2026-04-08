from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Iterable

MAGIC = b"AIPONG1\x00"
HEADER_BYTES = 64
RECORD_BYTES = 48
OBS_SCALE = 32767.0
REWARD_SCALE = 1000.0


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    yield from sorted(path.rglob("*.aipong"))


def parse_header(data: bytes) -> dict[str, int]:
    if len(data) < HEADER_BYTES:
        raise ValueError("file is smaller than the 64-byte header")
    if data[:8] != MAGIC:
        raise ValueError("invalid magic header")

    fields = struct.unpack_from("<HHHHIIIHHHHHHHHI", data, 8)
    header = {
        "version": fields[0],
        "header_bytes": fields[1],
        "record_bytes": fields[2],
        "sample_hz": fields[3],
        "sample_count": fields[4],
        "checkpoint_update": fields[5],
        "duration_ms": fields[6],
        "final_left_score": fields[7],
        "final_right_score": fields[8],
        "human_hits": fields[9],
        "bot_hits": fields[10],
        "max_rally": fields[11],
        "input_changes": fields[12],
        "visible_samples": fields[13],
        "flags": fields[14],
        "chunk_index": fields[15],
    }

    expected_size = header["header_bytes"] + header["sample_count"] * header["record_bytes"]
    if len(data) != expected_size:
        raise ValueError(f"expected {expected_size} bytes, got {len(data)}")
    if header["version"] != 1 or header["header_bytes"] != HEADER_BYTES or header["record_bytes"] != RECORD_BYTES:
        raise ValueError(f"unsupported header: {header}")

    return header


def parse_record(data: bytes, offset: int) -> dict:
    human_observation = [
        struct.unpack_from("<h", data, offset + index * 2)[0] / OBS_SCALE
        for index in range(9)
    ]
    bot_observation = [
        struct.unpack_from("<h", data, offset + 18 + index * 2)[0] / OBS_SCALE
        for index in range(9)
    ]

    return {
        "human_observation": human_observation,
        "bot_observation": bot_observation,
        "human_action": data[offset + 36],
        "bot_action": data[offset + 37],
        "human_reward": struct.unpack_from("<h", data, offset + 38)[0] / REWARD_SCALE,
        "bot_reward": struct.unpack_from("<h", data, offset + 40)[0] / REWARD_SCALE,
        "flags": data[offset + 42],
        "human_score": data[offset + 43],
        "bot_score": data[offset + 44],
        "rally_hits": struct.unpack_from("<H", data, offset + 45)[0],
        "visible": bool(data[offset + 47]),
    }


def decode_file(path: Path) -> tuple[dict, list[dict]]:
    data = path.read_bytes()
    header = parse_header(data)
    records = [
        parse_record(data, HEADER_BYTES + index * RECORD_BYTES)
        for index in range(header["sample_count"])
    ]
    return header, records


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode AI Pong .aipong binary trajectory files.")
    parser.add_argument("path", type=Path, help="A .aipong file or directory containing .aipong files.")
    parser.add_argument("--out", type=Path, help="Optional JSONL output path.")
    parser.add_argument("--summary", action="store_true", help="Only print file summaries.")
    args = parser.parse_args()

    files = list(iter_files(args.path))
    if not files:
        raise SystemExit(f"No .aipong files found under {args.path}")

    output_handle = args.out.open("w", encoding="utf-8") if args.out else None
    total_samples = 0

    try:
        for file_path in files:
            header, records = decode_file(file_path)
            total_samples += header["sample_count"]
            print(
                f"{file_path}: {header['sample_count']} samples, "
                f"checkpoint {header['checkpoint_update']}, "
                f"score {header['final_left_score']}:{header['final_right_score']}"
            )

            if output_handle and not args.summary:
                for record in records:
                    output_handle.write(
                        json.dumps(
                            {
                                "source_file": str(file_path),
                                "checkpoint_update": header["checkpoint_update"],
                                **record,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
    finally:
        if output_handle:
            output_handle.close()

    print(f"Decoded {len(files)} files and {total_samples} samples.")


if __name__ == "__main__":
    main()
