from dataclasses import dataclass
from typing import Dict, Optional

from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TaskDataloaders:
    train: DataLoader
    eval: DataLoader
    test: Optional[DataLoader] = None
    meta: Optional[Dict] = None


class DatasetBuilder:
    """Build decoupled dataloaders for a specific binary classification dataset."""

    name: str

    def build(self) -> TaskDataloaders:
        raise NotImplementedError
