from pathlib import Path
from typing import Dict, Type

from datasets.base import DatasetBuilder
from datasets.nlpdt import NLPDTDatasetBuilder


_BUILDERS: Dict[str, Type[DatasetBuilder]] = {
    NLPDTDatasetBuilder.name: NLPDTDatasetBuilder,
}


def get_builder(dataset_name: str, *, root_dir: Path, tokenizer) -> DatasetBuilder:
    builder_cls = _BUILDERS.get(dataset_name)
    if builder_cls is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {sorted(_BUILDERS)}")
    return builder_cls(root_dir=root_dir, tokenizer=tokenizer)
