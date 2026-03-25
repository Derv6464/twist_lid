import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
import omni.usd
from pxr import Usd, UsdPhysics, PhysxSchema

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
    usd_path = "/home/dgargan2/twist_lid/assets/bottle.usdc"

    stage = Usd.Stage.Open(usd_path)

    for prim in stage.Traverse():

        name = prim.GetName().lower()

        # apply physics to the bottle root
        if "cap" in name:
            print("Applying rigid body:", prim.GetPath())

            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(prim)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

        # apply SDF collider to the mesh
        if prim.GetTypeName() == "Mesh":
            print("Applying SDF collider:", prim.GetPath())

            collision_api = UsdPhysics.CollisionAPI.Apply(prim)

            meshCollision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            meshCollision.CreateApproximationAttr(PhysxSchema.Tokens.sdf)

            sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
            sdf_api.CreateSdfResolutionAttr(256)   
            sdf_api.CreateSdfMarginAttr(0.002)

    stage.GetRootLayer().Save()

def make_sdf_collision(sdf_prim_path):
    stage = omni.usd.get_context().get_stage()

    sdf_prim = stage.GetPrimAtPath(sdf_prim_path)
    UsdPhysics.CollisionAPI.Apply(sdf_prim)

    meshCollision = UsdPhysics.MeshCollisionAPI.Apply(sdf_prim)
    meshCollision.CreateApproximationAttr(PhysxSchema.Tokens.sdf)

    sdfMeshCollision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(sdf_prim)
    sdfMeshCollision.CreateSdfResolutionAttr(300)