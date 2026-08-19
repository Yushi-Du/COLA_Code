# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain configurations used by the three COLA training phases."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


# Phase 3 uses a strictly flat height field during behavior cloning.
FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.0,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0, 0),
            noise_step=0.02,
            border_width=0.25,
        )
    },
)


PHASE1_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "plain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.4,
            amplitude_range=(0.0, 0.001),
            num_waves=1.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1,
            noise_range=(-0.015, 0.015),
            noise_step=0.01,
            border_width=0.25,
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            inverted=False,
        ),
        "slope_inverted": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            inverted=True,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.1,
            amplitude_range=(0.0, 0.1),
            num_waves=5.0,
        ),
    },
)


# Phase 2 remains on the easier stair-free collaboration distribution.
ROUGH_EASY_TERRAINS_NO_STAIRS_EASY_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(-0.01, 0.01),
            noise_step=0.01,
            border_width=0.25,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.25,
            amplitude_range=(0.0, 0.1),
            num_waves=5.0,
        ),
        "plain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.5,
            amplitude_range=(0.0, 0.001),
            num_waves=1.0,
        ),
    },
)


__all__ = [
    "FLAT_TERRAINS_CFG",
    "PHASE1_TERRAINS_CFG",
    "ROUGH_EASY_TERRAINS_NO_STAIRS_EASY_CFG",
]
