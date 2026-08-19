"""Scene shared by fixed-bar teachers and students."""

from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR

from legged_lab.terrains.ray_caster_cfg import RayCasterCfg
from legged_lab.utils.env_utils.fixed_bar_attachment import FixedBarAttachmentCfg


@configclass
class CollaborationSceneCfg(InteractiveSceneCfg):
    """G1, terrain, sensors, carried bar, and wrist attachments."""

    def __init__(
        self,
        config: Any,
        bar_config: Any,
        physics_dt: float,
        step_dt: float,
        fixed_side: str,
    ):
        super().__init__(num_envs=config.num_envs, env_spacing=config.env_spacing)

        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type=config.terrain_type,
            terrain_generator=config.terrain_generator,
            max_init_terrain_level=config.max_init_terrain_level,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="max",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=(
                    f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
                    "TilesMarbleSpiderWhiteBrickBondHoned/"
                    "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
                ),
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        self.robot: ArticulationCfg = config.robot.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.carried_bar = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CarriedBar",
            spawn=sim_utils.CuboidCfg(
                size=bar_config.bar_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=bar_config.bar_mass),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="max",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.82, 0.55, 0.16), roughness=0.45
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=bar_config.initial_center_position,
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        self.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=True,
            update_period=physics_dt,
        )

        self.fixed_bar_attachments = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/FixedBarAttachments",
            spawn=FixedBarAttachmentCfg(
                fixed_side=fixed_side,
                left_bar_position=(0.0, 0.148661419, 0.0),
                right_bar_position=(0.0, -0.148651419, 0.0),
            ),
        )
        self.foot_balance_contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
            history_length=10,
            track_air_time=False,
            update_period=physics_dt,
        )

        self.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        self.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=(
                    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                    "kloofendal_43d_clear_puresky_4k.hdr"
                ),
            ),
        )

        if config.height_scanner.enable_height_scan:
            self.height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.height_scanner.prim_body_name,
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                attach_yaw_only=True,
                pattern_cfg=patterns.GridPatternCfg(
                    resolution=config.height_scanner.resolution,
                    size=config.height_scanner.size,
                ),
                debug_vis=config.height_scanner.debug_vis,
                mesh_prim_paths=["/World/ground"],
                update_period=step_dt,
                drift_range=config.height_scanner.drift_range,
            )


__all__ = ["CollaborationSceneCfg"]
