"""Unitree G1 asset variant with visual wrist connectors for a fixed bar."""

from copy import deepcopy

from legged_lab.assets import G1_DIRECT_FIXED_BAR_USD_PATH
from legged_lab.assets.unitree.unitree import G1_HAND_FIXED_CFG


G1_DIRECT_FIXED_BAR_CFG = deepcopy(G1_HAND_FIXED_CFG)
G1_DIRECT_FIXED_BAR_CFG.spawn.usd_path = G1_DIRECT_FIXED_BAR_USD_PATH
(
    G1_DIRECT_FIXED_BAR_CFG.spawn.articulation_props.solver_position_iteration_count
) = 8
(
    G1_DIRECT_FIXED_BAR_CFG.spawn.articulation_props.solver_velocity_iteration_count
) = 2


__all__ = [
    "G1_DIRECT_FIXED_BAR_CFG",
]
