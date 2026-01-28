
# patch_bottle_usd.py
# Usage:
#   ./isaaclab.sh -p patch_bottle_usd.py -- \
#       --in /abs/or/nucleus/path/to/bottle014.usd \
#       --out /abs/or/nucleus/path/to/bottle014_artic.usd

import argparse
from pxr import Usd, UsdPhysics, PhysxSchema, Sdf

def ensure_articulation_root(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"Prim not found: {prim_path}")
    if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        print(f"Applied ArticulationRootAPI to {prim_path}")
    else:
        print(f"ArticulationRootAPI already present on {prim_path}")
    # Optional: also apply PhysX articulation API
    if not prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        PhysxSchema.PhysxArticulationAPI.Apply(prim)
        print(f"Applied PhysxArticulationAPI to {prim_path}")

def constrain_d6_to_twist(stage, joint_path):
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim:
        raise RuntimeError(f"Joint prim not found: {joint_path}")

    d6 = UsdPhysics.Joint(joint_prim)  # generic joint interface
    if not d6:
        raise RuntimeError(f"Prim at {joint_path} is not a Physics Joint")

    # Lock linear X/Y/Z and RotX/RotY; free RotZ (twist).
    # For D6 in USD, axes are: "transX/Y/Z" and "rotX/Y/Z".
    # Lock = 0, Limited = 1, Free = 2
    def set_motion(axis, mode):
        attr = joint_prim.GetAttribute(f"physics:motion:{axis}")
        if not attr:
            attr = joint_prim.CreateAttribute(f"physics:motion:{axis}", Sdf.ValueTypeNames.Token)
        attr.Set(mode)

    for axis in ("transX", "transY", "transZ", "rotX", "rotY"):
        set_motion(axis, "locked")
    set_motion("rotZ", "free")

    # Optional: limit the twist (rotZ) range like a revolute:
    #   e.g., -360..360 deg
    def set_limit(axis, low, high):
        la = joint_prim.GetAttribute(f"physics:lowerLimit:{axis}") or joint_prim.CreateAttribute(
            f"physics:lowerLimit:{axis}", Sdf.ValueTypeNames.Double)
        ha = joint_prim.GetAttribute(f"physics:upperLimit:{axis}") or joint_prim.CreateAttribute(
            f"physics:upperLimit:{axis}", Sdf.ValueTypeNames.Double)
        la.Set(low)
        ha.Set(high)
        # Also set motion to 'limited'
        set_motion(axis, "limited")

    # Comment out if you truly want 'free' rotation
    # set_limit("rotZ", -6.283185307179586, 6.283185307179586)  # +/- 360 deg in radians

    print(f"Constrained D6 joint at {joint_path} to only twist (rotZ).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--root", default="/bottle014", help="Articulation root prim inside file")
    ap.add_argument("--joint", default="/bottle014/lid/D6Joint", help="D6 joint prim path")
    args = ap.parse_args()

    stage = Usd.Stage.Open(args.in_path)
    if stage is None:
        raise RuntimeError(f"Failed to open {args.in_path}")

    ensure_articulation_root(stage, args.root)
    constrain_d6_to_twist(stage, args.joint)

    stage.GetRootLayer().Export(args.out_path)
    print(f"Saved patched USD to: {args.out_path}")

if __name__ == "__main__":
    main()
