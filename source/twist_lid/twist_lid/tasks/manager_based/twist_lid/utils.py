def parse_usd_and_create_subassets(usd_path, env_cfg):
    stage = get_stage(usd_path)
    prims = get_all_prims(stage)

    for prim in prims:

        if not is_rigidbody(prim):
            continue

        prim_path = prim.GetPath().pathString
        name = prim_path.split("/")[-1].lower()

        # Only grab the two things we care about
        if name not in ["body", "lid"]:
            continue

        pos, rot = get_prim_pos_rot(prim)

        # Rename "body" -> bottle
        if name == "body":
            scene_name = "bottle"
        else:
            scene_name = "lid"

        new_prim_path = f"{{ENV_REGEX_NS}}/Scene/{scene_name}"

        rigidcfg = RigidObjectCfg(
            prim_path=new_prim_path,
            spawn=RigidObjectSpawnerCfg(func=spawn_from_prim_path),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=pos,
                rot=rot,
            ),
        )

        setattr(env_cfg.scene, scene_name, rigidcfg)