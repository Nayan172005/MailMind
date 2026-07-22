"""Streaming sampling utilities for cleaned email datasets."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import pandas as pd

try:
    from .cleaner import PUBLIC_COLUMNS
    from .config import PreprocessingConfig
except ImportError:  # pragma: no cover - supports direct script execution.
    from cleaner import PUBLIC_COLUMNS
    from config import PreprocessingConfig


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Reservoir:
    """Fixed-size deterministic reservoir sample."""

    size: int
    random_state: int
    rows_seen: int = 0
    rows: list[dict[str, object]] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the random number generator after dataclass creation."""

        self._rng = random.Random(self.random_state)

    def update(self, frame: pd.DataFrame) -> None:
        """Update the reservoir with one cleaned chunk."""

        if frame.empty:
            return

        for row in frame.to_dict(orient="records"):
            self.rows_seen += 1
            if len(self.rows) < self.size:
                self.rows.append(row)
                continue

            replacement_index = self._rng.randrange(self.rows_seen)
            if replacement_index < self.size:
                self.rows[replacement_index] = row

    def to_dataframe(self) -> pd.DataFrame:
        """Return the current sample as a DataFrame."""

        if not self.rows:
            return pd.DataFrame(columns=PUBLIC_COLUMNS)
        return pd.DataFrame.from_records(self.rows)


class DatasetSampler:
    """Generate deterministic Parquet samples without loading all rows."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self._reservoirs = {
            size: Reservoir(size=size, random_state=config.random_state)
            for size in config.sample_sizes
        }

    def update(self, frame: pd.DataFrame) -> None:
        """Feed one cleaned chunk into all configured samplers."""

        for reservoir in self._reservoirs.values():
            reservoir.update(frame)

    def write_samples(self) -> dict[int, int]:
        """Write configured sample datasets.

        Returns:
            Mapping of requested sample size to actual rows written.
        """

        written: dict[int, int] = {}
        for size, reservoir in self._reservoirs.items():
            output_path = self.config.sample_path(size)
            sample = reservoir.to_dataframe()

            if len(sample) < size:
                logger.warning(
                    "Requested sample size %s but cleaned dataset has only %s rows",
                    size,
                    len(sample),
                )

            sample.to_parquet(output_path, index=False, engine="pyarrow")
            written[size] = len(sample)
            logger.info("Wrote sample dataset: %s (%s rows)", output_path, len(sample))

        return written
