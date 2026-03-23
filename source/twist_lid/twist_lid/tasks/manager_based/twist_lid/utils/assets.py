import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

KITCHEN_W_BOTTLE_PATH = '/home/dgargan2/twist_lid/assets/Lightwheel_OpenSource/Locomotion/KitchenRoom/Collected_waterbottle_7/waterbottle_7.usd'
BOTTLE_ONLY_PATH = '/home/dgargan2/bottle_open/assets/bottle.usda'

KITCHEN_W_BOTTLE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=KITCHEN_W_BOTTLE_PATH
    )
)

BOTTLE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=BOTTLE_ONLY_PATH
    )
)

def make_isacc_compatible():
    from pxr import Usd, UsdPhysics, PhysxSchema

    usd_path = "/home/dgargan2/twist_lid/assets/bottle.usdz"

    stage = Usd.Stage.Open(usd_path)

    for prim in stage.Traverse():

        name = prim.GetName().lower()

        # apply physics to the bottle root
        if "bottle" in name:
            print("Applying rigid body:", prim.GetPath())

            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(prim)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

        # apply SDF collider to the mesh
        if prim.GetTypeName() == "Mesh":
            print("Applying SDF collider:", prim.GetPath())

            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)

            # enable SDF collision
            sdf_api.CreateSdfResolutionAttr(256)   # higher = more accurate
            sdf_api.CreateSdfMarginAttr(0.002)     # small safety margin

    stage.GetRootLayer().Save()

    print("Done - object is now rigid body with SDF collision")