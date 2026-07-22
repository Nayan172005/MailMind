"""Run the SpamAssassin preprocessing pipeline."""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from datasets.preprocessing.spamassassin.main import SpamAssassinPreprocessingPipeline
from datasets.preprocessing.spamassassin.config import SpamAssassinConfig
from datasets.preprocessing.spamassassin.utils import setup_logging

setup_logging("INFO")
config = SpamAssassinConfig.default()
SpamAssassinPreprocessingPipeline(config).run()
