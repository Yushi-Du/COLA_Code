"""USD joint spawner for the G1 direct fixed-bar experiment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pxr import Gf, Sdf, Usd, UsdPhysics

from isaaclab.sim import SpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage, standardize_xform_ops
from isaaclab.utils import configclass


def _quat(value: tuple[float, float, float, float]) -> Gf.Quatf:
    return Gf.Quatf(value[0], Gf.Vec3f(*value[1:]))


@clone
def spawn_fixed_bar_attachment(
    prim_path: str,
    cfg: "FixedBarAttachmentCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a rigid left wrist weld and a translation-only right closure."""

    del translation, orientation, kwargs
    stage = get_current_stage()
    root = stage.DefinePrim(prim_path, "Xform")
    standardize_xform_ops(root)
    env_path = prim_path.rsplit("/", 1)[0]
    bar_path = Sdf.Path(f"{env_path}/{cfg.bar_prim_name}")
    left_wrist_path = Sdf.Path(
        f"{env_path}/{cfg.robot_prim_name}/{cfg.left_wrist_body_name}"
    )
    right_wrist_path = Sdf.Path(
        f"{env_path}/{cfg.robot_prim_name}/{cfg.right_wrist_body_name}"
    )

    if cfg.fixed_side == "left":
        fixed_name = "LeftFixedJoint"
        fixed_wrist_path = left_wrist_path
        fixed_wrist_position = cfg.left_wrist_position
        fixed_wrist_rotation = cfg.left_wrist_rotation
        fixed_bar_position = cfg.left_bar_position
        point_name = "RightPointJoint"
        point_wrist_path = right_wrist_path
        point_wrist_position = cfg.right_wrist_position
        point_bar_position = cfg.right_bar_position
    elif cfg.fixed_side == "right":
        fixed_name = "RightFixedJoint"
        fixed_wrist_path = right_wrist_path
        fixed_wrist_position = cfg.right_wrist_position
        fixed_wrist_rotation = cfg.right_wrist_rotation
        fixed_bar_position = cfg.right_bar_position
        point_name = "LeftPointJoint"
        point_wrist_path = left_wrist_path
        point_wrist_position = cfg.left_wrist_position
        point_bar_position = cfg.left_bar_position
    else:
        raise ValueError(f"unsupported fixed side: {cfg.fixed_side!r}")

    fixed_joint = UsdPhysics.FixedJoint.Define(
        stage, f"{prim_path}/{fixed_name}"
    )
    fixed_joint.CreateBody0Rel().SetTargets([fixed_wrist_path])
    fixed_joint.CreateBody1Rel().SetTargets([bar_path])
    fixed_joint.CreateLocalPos0Attr(Gf.Vec3f(*fixed_wrist_position))
    fixed_joint.CreateLocalRot0Attr(_quat(fixed_wrist_rotation))
    fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(*fixed_bar_position))
    fixed_joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    fixed_joint.CreateExcludeFromArticulationAttr(True)

    point_joint = UsdPhysics.SphericalJoint.Define(
        stage, f"{prim_path}/{point_name}"
    )
    point_joint.CreateBody0Rel().SetTargets([point_wrist_path])
    point_joint.CreateBody1Rel().SetTargets([bar_path])
    point_joint.CreateLocalPos0Attr(Gf.Vec3f(*point_wrist_position))
    point_joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    point_joint.CreateLocalPos1Attr(Gf.Vec3f(*point_bar_position))
    point_joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    point_joint.CreateExcludeFromArticulationAttr(True)
    return root


@configclass
class FixedBarAttachmentCfg(SpawnerCfg):
    """Canonical attachment frames with a selectable welded wrist."""

    func: Callable = spawn_fixed_bar_attachment
    robot_prim_name: str = "Robot"
    bar_prim_name: str = "CarriedBar"
    left_wrist_body_name: str = "left_wrist_yaw_link"
    right_wrist_body_name: str = "right_wrist_yaw_link"
    fixed_side: Literal["left", "right"] = "left"
    left_wrist_position: tuple[float, float, float] = (
        0.098224133,
        -0.028772614,
        0.000020504,
    )
    right_wrist_position: tuple[float, float, float] = (
        0.098224133,
        0.028772614,
        0.000020504,
    )
    left_bar_position: tuple[float, float, float] = (0.0, 0.561661419, 0.0)
    right_bar_position: tuple[float, float, float] = (0.0, 0.264348581, 0.0)
    left_wrist_rotation: tuple[float, float, float, float] = (
        0.707127478,
        0.707086077,
        -0.000087163,
        0.000048312,
    )
    right_wrist_rotation: tuple[float, float, float, float] = (
        0.707127478,
        -0.707086077,
        -0.000087163,
        -0.000048312,
    )


__all__ = ["FixedBarAttachmentCfg", "spawn_fixed_bar_attachment"]
