"""Training entry point.

    python -m minimegatron.train --config configs/tiny.yaml
    python -m minimegatron.train --config configs/tiny.yaml --resume runs/tiny/ckpt_000500.pt
    python -m minimegatron.train --config configs/tiny.yaml --set train.max_steps=20
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import yaml

from minimegatron.config import ConfigError, load_yaml
from minimegatron.distributed.setup import destroy_distributed, init_distributed
from minimegatron.experiment import config_from_dict
from minimegatron.training.trainer import Trainer


def parse_override(text: str) -> tuple[list[str], Any]:
    """Parse `section.key=value`. The value goes through the YAML parser, so `20` is an int,
    `true` is a bool and `null` is None -- exactly as the same text in a config file would be."""
    if "=" not in text:
        raise ConfigError(f"--set expects section.key=value, got {text!r}")
    dotted, _, raw = text.partition("=")
    path = [part for part in dotted.strip().split(".") if part]
    if len(path) < 2:
        raise ConfigError(f"--set key must be dotted, like train.max_steps, got {dotted!r}")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ConfigError(f"--set {text!r}: could not parse value: {e}") from e
    return path, value


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dotted overrides to a raw config mapping, before validation.

    Overriding the raw mapping rather than the built dataclasses means an override is validated by
    the same code path as the file: a typo in `--set` is rejected with the same 'did you mean'
    hint, instead of silently setting an attribute nobody reads.
    """
    out = {section: dict(value) for section, value in raw.items() if isinstance(value, dict)}
    out.update({k: v for k, v in raw.items() if not isinstance(v, dict)})
    for text in overrides:
        path, value = parse_override(text)
        if len(path) > 2:
            raise ConfigError(f"--set supports section.key, got {'.'.join(path)!r}")
        section, key = path
        out.setdefault(section, {})
        if not isinstance(out[section], dict):
            raise ConfigError(f"--set {text!r}: section {section!r} is not a mapping")
        out[section][key] = value
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minimegatron.train", description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML experiment config")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--out-dir", default=None, help="override train.out_dir")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="section.key=value",
        help="override a config value; repeatable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        raw = apply_overrides(load_yaml(args.config), args.overrides)
        cfg = config_from_dict(raw, source=args.config)
    except ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 2

    # No-op without torchrun's environment, so the same entry point serves single-process and
    # distributed runs. Teardown goes in a `finally`: a rank that exits while its peers are inside
    # a collective leaves them blocked until the timeout, turning one failure into a hung job.
    dist_info = init_distributed(cfg.distributed)
    try:
        trainer = Trainer(cfg, dist_info=dist_info, out_dir=args.out_dir, echo=dist_info.is_main)
        if dist_info.is_main:
            world = (
                f" | ranks: {dist_info.world_size}" if dist_info.is_distributed else ""
            )
            print(
                f"model: {trainer.raw_model.num_params():,} parameters "
                f"({trainer.raw_model.num_params(include_embeddings=False):,} non-embedding) | "
                f"device: {trainer.device} | steps: {cfg.train.max_steps}{world}",
                flush=True,
            )
        if args.resume:
            meta = trainer.resume(args.resume)
            if dist_info.is_main:
                print(f"resumed from {args.resume} at step {meta['step']}", flush=True)

        trainer.fit()
    finally:
        destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
