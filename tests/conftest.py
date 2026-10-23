from pathlib import Path

import pytest

from minimegatron.model.config import ModelConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def nano_config() -> ModelConfig:
    """A model small enough that a full forward/backward takes milliseconds on CPU."""
    return ModelConfig(vocab_size=128, seq_len=16, n_layers=2, n_heads=4, d_model=32, d_ff=64)
