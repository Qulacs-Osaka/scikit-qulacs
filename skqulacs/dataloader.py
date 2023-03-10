from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


@dataclass
class DataLoader:
    """
    Data loader. This class is an iterator that yields mini-batches.
    Their size is specified by `batch_size` argument.
    You can specify whether to shuffle the data or not by `shuffle` argument.
    """

    x: NDArray[np.float_]
    y: NDArray[np.float_]
    batch_size: int = 1
    shuffle: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("x and y must have the same length.")

    def __iter__(self) -> "_DataLoaderIterator":
        return _DataLoaderIterator(self)

    def __len__(self) -> int:
        return (len(self.x) + self.batch_size - 1) // self.batch_size


@dataclass
class _DataLoaderIterator:
    loader: DataLoader
    rng: Generator = field(init=False)
    indices: list[int] = field(init=False, default_factory=list)
    current_index: int = field(init=False, default=0)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.loader.seed)
        self.indices = list(range(len(self.loader.x)))
        if self.loader.shuffle:
            self.rng.shuffle(self.indices)

    def __next__(self):
        if self.current_index >= len(self.indices):
            raise StopIteration

        selected = self.indices[
            self.current_index : (self.current_index + self.loader.batch_size)
        ]
        self.current_index += self.loader.batch_size
        return self.loader.x[selected], self.loader.y[selected]
