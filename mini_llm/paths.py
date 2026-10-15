"""Where things live on disk.

Every path is derived from the repo root rather than the current working
directory, so a script behaves the same no matter where it is run from.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
VERDICT_FILE = DATA_DIR / "the_verdict.txt"
SPAM_DIR = DATA_DIR / "spam"

PLOTS_DIR = PROJECT_ROOT / "plots"
MODELS_DIR = PROJECT_ROOT / "saved_models"
# where the downloaded OpenAI GPT-2 checkpoints go
GPT2_DIR = PROJECT_ROOT / "gpt2"
