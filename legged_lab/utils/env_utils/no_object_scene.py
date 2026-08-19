"""Pure fixed-hand G1 scene for the static no-object population."""

from typing import Any

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from legged_lab.utils.env_utils.locomotion_scene import (
    LocomotionSceneCfg,
)


@configclass
class NoObjectSceneCfg(LocomotionSceneCfg):
    """Locomotion scene plus the ten-frame foot-load sensor used by COLA."""

    def __init__(self, config: Any, physics_dt: float, step_dt: float):
        super().__init__(config, physics_dt, step_dt)
        self.foot_balance_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
            history_length=10,
            track_air_time=False,
            update_period=physics_dt,
        )


__all__ = ["NoObjectSceneCfg"]
