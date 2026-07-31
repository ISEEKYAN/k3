#!/usr/bin/env python3
"""Convert an existing GSM8K SFT parquet into VERL's rule-reward schema."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any


FINAL_ANSWER = re.compile(r"#### (\-?[0-9\.,]+)")


def convert_messages(
    messages: list[dict[str, str]],
    *,
    split: str,
    index: int,
) -> dict[str, Any]:
    if (
        len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise ValueError(f"{split}[{index}] is not one user/assistant exchange")

    question = messages[0]["content"]
    answer = messages[1]["content"]
    match = FINAL_ANSWER.search(answer)
    if match is None:
        raise ValueError(f"{split}[{index}] has no GSM8K final answer")
    ground_truth = match.group(1).replace(",", "")

    return {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": index,
            "answer": answer,
            "question": question,
        },
    }


def convert_split(source: Path, destination: Path, split: str) -> int:
    import datasets

    dataset = datasets.load_dataset(
        "parquet",
        data_files={split: str(source)},
        split=split,
    )
    converted = dataset.map(
        lambda row, index: convert_messages(
            row["messages"],
            split=split,
            index=index,
        ),
        with_indices=True,
        remove_columns=dataset.column_names,
    )
    converted.to_parquet(str(destination))
    return len(converted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts = {
        split: convert_split(
            args.source_dir / f"{split}.parquet",
            args.output_dir / f"{split}.parquet",
            split,
        )
        for split in ("train", "test")
    }
    print(
        "K3_GSM8K_RL_DATA_OK",
        f"train_rows={counts['train']}",
        f"test_rows={counts['test']}",
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
