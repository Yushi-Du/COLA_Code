"""Paths for the assets shipped with the COLA release."""

from pathlib import Path


COLA_ROOT = Path(__file__).resolve().parents[2]
COLA_ASSET_DIR = COLA_ROOT / "assets"
G1_FIXED_HAND_USD_PATH = str(
    COLA_ASSET_DIR
    / "unitree_g1"
    / "g1_29dof_with_hand_rev_1_0_hand_fixed_for_collab.usd"
)
G1_DIRECT_FIXED_BAR_USD_PATH = str(
    COLA_ASSET_DIR
    / "unitree_g1"
    / "g1_29dof_direct_fixed_bar.usda"
)
