# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast sensor."""


from isaaclab.utils import configclass

from isaaclab.sensors.ray_caster import RayCasterCfg as BaseRayCasterCfg
from .ray_caster import RayCaster


@configclass
class RayCasterCfg(BaseRayCasterCfg):

    class_type: type = RayCaster
