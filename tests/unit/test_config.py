import dataclasses
import re

import pytest

from minimegatron.config import ConfigError, from_dict
from minimegatron.experiment import load_config
from minimegatron.model.config import ModelConfig
from tests.conftest import REPO_ROOT

BASE = dict(n_layers=2, n_heads=4, d_model=32)


# --- presets and parameter counts ---------------------------------------------------------------


@pytest.mark.parametrize(
    "preset, expected",
    [("tiny", 11_487_360), ("small", 51_237_888), ("medium", 124_475_904)],
)
def test_preset_param_counts(preset, expected):
    cfg = load_config(REPO_ROOT / "configs" / f"{preset}.yaml").model
    assert cfg.num_params() == expected


def test_param_formula_matches_gpt2_small():
    # GPT-2 small (unpadded vocab) has a published count of 124,439,808.
    cfg = ModelConfig(n_layers=12, n_heads=12, d_model=768, vocab_size=50257, seq_len=1024)
    assert cfg.num_params() == 124_439_808


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, 21_760),  # layernorm, bias, tied
        ({"norm_type": "rmsnorm"}, 21_600),  # norms lose beta: 5 norms x 32
        ({"tie_embeddings": False}, 25_856),  # separate 128 x 32 LM head
        ({"bias": False}, 21_152),  # no linear biases, no LayerNorm beta
    ],
)
def test_param_formula_variants(nano_config, overrides, expected):
    cfg = dataclasses.replace(nano_config, **overrides)
    assert cfg.num_params() == expected


# --- defaults and derived values ----------------------------------------------------------------


def test_d_ff_defaults_to_4x_d_model():
    assert ModelConfig(**BASE).d_ff == 128


def test_head_dim():
    assert ModelConfig(n_layers=1, n_heads=4, d_model=192).head_dim == 48


def test_config_is_frozen(nano_config):
    with pytest.raises(dataclasses.FrozenInstanceError):
        nano_config.n_layers = 3


# --- validation ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"d_model": 30}, "divisible by n_heads"),
        ({"n_layers": 0}, "n_layers must be a positive integer"),
        ({"vocab_size": -1}, "vocab_size must be a positive integer"),
        ({"n_heads": True}, "n_heads must be a positive integer"),  # bool is an int subclass
        ({"d_model": 32.0}, "d_model must be a positive integer"),
        ({"dropout": 1.0}, "dropout must be in [0, 1)"),
        ({"dropout": "0.1"}, "dropout must be a number"),
        ({"bias": 1}, "bias must be true or false"),
        ({"attention_backend": "flash"}, "attention_backend must be one of"),
        ({"norm_type": "batchnorm"}, "norm_type must be one of"),
        # dtype is parameter storage; bf16 compute is train.precision (Milestone 3).
        ({"dtype": "bfloat16"}, "train.precision: bf16"),
        ({"dtype": "float16"}, "dtype must be one of"),
    ],
)
def test_invalid_values_rejected(overrides, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        ModelConfig(**{**BASE, **overrides})


def test_all_errors_reported_at_once():
    with pytest.raises(ConfigError) as exc:
        ModelConfig(n_layers=0, n_heads=4, d_model=30, norm_type="batchnorm")
    msg = str(exc.value)
    assert "n_layers" in msg and "divisible" in msg and "norm_type" in msg


# --- dict and YAML loading ----------------------------------------------------------------------


def test_from_dict_unknown_key_suggests_fix():
    with pytest.raises(ConfigError, match="unknown key 'n_layer'.*did you mean 'n_layers'"):
        from_dict(ModelConfig, {**BASE, "n_layer": 4}, section="model")


def test_from_dict_missing_required_key():
    with pytest.raises(ConfigError, match="missing required key.*d_model"):
        from_dict(ModelConfig, {"n_layers": 2, "n_heads": 4}, section="model")


def test_from_dict_rejects_non_mapping():
    with pytest.raises(ConfigError, match="must be a mapping"):
        from_dict(ModelConfig, [1, 2, 3], section="model")


def test_load_config_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_malformed_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model: [unclosed\n")
    with pytest.raises(ConfigError, match="could not parse"):
        load_config(path)


def test_load_config_unknown_section(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model: {n_layers: 2, n_heads: 4, d_model: 32}\nmodle: {}\n")
    with pytest.raises(ConfigError, match="unknown section"):
        load_config(path)


def test_load_config_missing_model_section(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("")
    with pytest.raises(ConfigError, match="missing required section 'model'"):
        load_config(path)
