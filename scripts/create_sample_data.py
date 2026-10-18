"""Create lightweight sample datasets for repository testing and zero-setup demo.

Extracts an anonymized, small slice of tenders and contractor profiles so any
user cloning the repository can immediately execute EDA, train models, and
run the Streamlit simulator without downloading heavy raw datasets.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import DATA_DIR, SAMPLE_PROFILES_PATH, SAMPLE_TENDERS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Search locations for full processed parquets
CANDIDATE_DIRS = [
    Path(r"C:\Users\User\Documents\InferIA"),
    DATA_DIR,
    PROJECT_ROOT / "data",
]


def create_sample_datasets(sample_size: int = 500, n_profiles: int = 250) -> None:
    """Generate sample_tenders.parquet and sample_profiles.parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    test_source = None
    profiles_source = None

    for d in CANDIDATE_DIRS:
        t_path = d / "test_procesado_v2_limpio.parquet"
        p_path = d / "perfiles_empresas.parquet"
        if t_path.exists() and test_source is None:
            test_source = t_path
        if p_path.exists() and profiles_source is None:
            profiles_source = p_path

    if test_source is not None:
        logger.info("Extracting %d tenders from %s...", sample_size, test_source)
        df_tenders = pd.read_parquet(test_source)
        sample_df = df_tenders.sample(min(sample_size, len(df_tenders)), random_state=42)
        sample_df.to_parquet(SAMPLE_TENDERS_PATH, index=False)
        logger.info("Saved sample tenders to: %s (%d rows)", SAMPLE_TENDERS_PATH, len(sample_df))
    else:
        logger.warning("No full tender parquet found to extract sample from.")

    if profiles_source is not None:
        logger.info("Extracting %d contractor profiles from %s...", n_profiles, profiles_source)
        df_prof = pd.read_parquet(profiles_source)
        sample_prof = df_prof.head(min(n_profiles, len(df_prof)))
        sample_prof.to_parquet(SAMPLE_PROFILES_PATH)
        logger.info("Saved sample profiles to: %s (%d rows)", SAMPLE_PROFILES_PATH, len(sample_prof))
    else:
        logger.warning("No full profiles parquet found to extract sample from.")


if __name__ == "__main__":
    create_sample_datasets()
