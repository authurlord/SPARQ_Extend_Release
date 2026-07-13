"""Local HybridQA raw-data loader.

This avoids requiring a prebuilt Hugging Face datasets cache. It reads the
official HybridQA released_data JSON files and WikiTables-WithLinks zip archive
directly.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List


SPLIT_TO_FILE = {
    "train": "train.json",
    "validation": "dev.json",
    "dev": "dev.json",
    "test": "test.json",
}


class LocalHybridQADataset:
    def __init__(self, raw_dir: Path, split: str) -> None:
        self.raw_dir = Path(raw_dir)
        split_file = SPLIT_TO_FILE.get(split, f"{split}.json")
        qa_path = self.raw_dir / split_file
        if not qa_path.exists():
            raise FileNotFoundError(f"Missing HybridQA split file: {qa_path}")

        zip_paths = sorted(self.raw_dir.glob("WikiTables-WithLinks*.zip"))
        if not zip_paths:
            raise FileNotFoundError(
                f"Missing WikiTables-WithLinks zip in {self.raw_dir}. "
                "Expected a file named WikiTables-WithLinks*.zip."
            )

        self.examples: List[Dict[str, Any]] = json.loads(qa_path.read_text())
        self.zip_path = zip_paths[0]
        self._zip = zipfile.ZipFile(self.zip_path)
        names = self._zip.namelist()
        roots = [n.split("/", 1)[0] for n in names if "/" in n]
        self.root = roots[0] if roots else ""

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = dict(self.examples[idx])
        table_id = example["table_id"]
        table = self._read_json(f"{self.root}/tables_tok/{table_id}.json")
        url_data = self._read_json(f"{self.root}/request_tok/{table_id}.json")

        table["header"] = [header[0] if isinstance(header, list) else header for header in table["header"]]
        rows = []
        for row in table["data"]:
            for cell in row:
                value = cell[0]
                urls = cell[1]
                rows.append(
                    {
                        "value": value,
                        "urls": [
                            {"url": url, "summary": url_data.get(url, "")}
                            for url in urls
                        ],
                    }
                )
        table["data"] = rows

        if "answer-text" in example and "answer_text" not in example:
            example["answer_text"] = example.pop("answer-text")
        example.setdefault("answer_text", "")
        example["table"] = table
        return example

    def _read_json(self, name: str) -> Any:
        with self._zip.open(name) as f:
            return json.loads(f.read().decode("utf-8"))


def load_hybridqa_dataset(split: str, raw_dir: Path | None = None):
    if raw_dir is not None:
        return LocalHybridQADataset(raw_dir, split)

    from datasets import load_dataset

    return load_dataset("hybrid_qa", split=split)
