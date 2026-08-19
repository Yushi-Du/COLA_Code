#!/usr/bin/env python3
"""Export a COLA phase-1 actor or phase-3 student as TorchScript."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    prefix = "actor." if any(key.startswith("actor.") for key in state) else "student."
    actor_state = {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not actor_state:
        raise RuntimeError("checkpoint contains no actor.* or student.* parameters")

    linear_indices = sorted(
        int(key.split(".", 1)[0])
        for key in actor_state
        if key.endswith(".weight")
    )
    if not linear_indices:
        raise RuntimeError("actor contains no linear layers")

    layers: list[nn.Module] = []
    for position, index in enumerate(linear_indices):
        weight = actor_state[f"{index}.weight"]
        layers.append(nn.Linear(weight.shape[1], weight.shape[0]))
        if position != len(linear_indices) - 1:
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers)
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()

    input_width = actor_state[f"{linear_indices[0]}.weight"].shape[1]
    example = torch.zeros(1, input_width)
    with torch.inference_mode():
        policy = torch.jit.trace(actor, example)
        output = policy(example)
    if output.shape != (1, 29):
        raise RuntimeError(f"expected a (1, 29) actor output, received {output.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy.save(str(args.output))
    print(
        f"POLICY_EXPORT_PASS source={prefix.removesuffix('.')} "
        f"input={input_width} output=29 path={args.output}"
    )


if __name__ == "__main__":
    main()
