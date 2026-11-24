"""Example: physically generate real-style W1 RGB-D data for VLA training.

One GPU worker at a time. Failed candidates are retained with success=false
and are never converted into the accepted training HDF5 directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import yaml
from w1_sim_interface import W1Simulation,ROOT
from convert_w1_native import convert,validate
from analyze_w1_motion import analyze as analyze_motion


def snapshot(config,destination):
    cfg_path=ROOT/f"env_cfg/task_config/{config}.yml";cfg=yaml.safe_load(cfg_path.read_text())
    paths=[cfg_path,ROOT/cfg["asset_manifest"],ROOT/"envs/w1_finals_real.py",ROOT/"envs/w1_finals.py",
           ROOT/"envs/w1_chili_pot.py",ROOT/"envs/utils/finals_assets.py",ROOT/"envs/robot/robot.py",
           ROOT/"envs/robot/planner.py",ROOT/"envs/robot/ik.py",ROOT/"scripts/w1_sim_interface.py",
           ROOT/"envs/robot/curobo_compat.py",
           ROOT/"scripts/w1_recording.py",ROOT/"scripts/convert_w1_native.py",Path(__file__).resolve(),
           ROOT/"scripts/w1_stage_instructions.py", ROOT/"scripts/w1_scene_variation.py",
           ROOT/"scripts/w1_task_monitor.py",ROOT/"envs/_base_task.py",
           ROOT/"scripts/w1_collision_quality.py",ROOT/"envs/robot/cartesian_trajectory.py",
           ROOT/"scripts/w1_stir_monitor.py",ROOT/"scripts/prepare_w1_back_up_reference.py",
           ROOT/"scripts/w1_motion_quality.py",ROOT/"scripts/analyze_w1_motion.py",
           ROOT/"scripts/w1_dual_control.py",ROOT/"scripts/w1_dual_collision.py",ROOT/"scripts/w1_dual_trajectory.py",
           ROOT/"env_cfg/task_config/_camera_config.yml",ROOT/"env_cfg/task_config/_embodiment_config.yml",
           ROOT/"scripts/prepare_w1_real_reference.py",
           ROOT/"scripts/prepare_w1_continuous_reference.py",ROOT/"scripts/prepare_w1_image_reference.py",
           ROOT/"env_cfg/w1_real_image_alignment_v1.json",
           ROOT/"env_cfg/w1_wrist_image_fit_v1.json",
           ROOT/"w1_urdf/urdf/play_g2.urdf",
           ROOT/"envs/camera/camera.py",ROOT/"XPolicyLab/utils/process_data.py"]
    registry=yaml.safe_load((ROOT/"env_cfg/task_config/_embodiment_config.yml").read_text())
    embodiment=ROOT/registry[cfg["embodiment"][0]]["file_path"]
    paths+=[p for p in embodiment.glob("*.yml") if p.name!="curobo.yml"]
    if cfg.get("scene_variation"):
        paths.append(ROOT / cfg["scene_variation"]["camera_profiles"])
    for field in ('camera_reference_file','native_meta_template'):
        if cfg.get(field):paths.append(ROOT/cfg[field])
    for optional in (ROOT/'episode_0001.zip',ROOT/cfg['real_episode']/'meta.json',ROOT/'w1_replay_urdf/w1_replay_urdf/urdf/replay.urdf'):
        if optional.exists():paths.append(optional)
    urdfs=list(embodiment.glob("*.urdf"));paths+=urdfs
    for file in urdfs:
        for mesh in ET.parse(file).getroot().iter("mesh"):paths.append((file.parent/mesh.get("filename")).resolve())
    assets=yaml.safe_load((ROOT/cfg["asset_manifest"]).read_text())
    for asset in assets.values():
        for key in ("visual","collision","metadata"):
            if key in asset:paths.append(ROOT/asset[key])
        if "metadata" in asset:
            for part in json.loads((ROOT/asset["metadata"]).read_text())["parts"]:
                if "collision_mesh" in part:paths.append(ROOT/part["collision_mesh"])
    destination.mkdir(parents=True,exist_ok=True)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
    (destination/"provenance.json").write_text(json.dumps({"sha256":hashes,"config":config},indent=2)+"\n")
    shutil.copy2(cfg_path,destination/"task_config.yml");shutil.copy2(ROOT/cfg["asset_manifest"],destination/"asset_manifest.yml")
    return hashes


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",default="w1_finals_real_v7_diverse")
    p.add_argument("--output",type=Path,default=ROOT/"data/w1_real_style_v1")
    p.add_argument("--episodes",type=int,default=1);p.add_argument("--max-attempts",type=int,default=12)
    p.add_argument("--start-seed",type=int,default=0)
    args=p.parse_args()
    if args.output.exists():p.error("Choose a fresh output directory to preserve provenance")
    hashes=snapshot(args.config,args.output)
    accepted=[];attempts=[]
    for seed in range(args.start_seed,args.start_seed+args.max_attempts):
        if len(accepted)>=args.episodes:break
        interface=W1Simulation(args.config);directory=args.output/f"candidates/seed_{seed:04d}"
        try:
            interface.reset(seed,record_dir=directory,episode_id=len(accepted))
            info,success=interface.expert()
            if not success:raise RuntimeError("Full physical task success checks failed")
            task=interface.task
            if task.minimal_rotation:
                quality=analyze_motion(directory)
                (directory/'motion_quality.json').write_text(json.dumps(quality,indent=2)+'\n')
                info['motion_quality_checks']=quality['checks']
                (directory/'scene_info.json').write_text(json.dumps(info,indent=2)+'\n')
                meta=json.loads((directory/'meta.json').read_text())
                meta['extra']['motion_quality_passed']=quality['success']
                meta['extra']['success']=bool(success and quality['success'])
                (directory/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
                if not quality['success']:raise RuntimeError(f"EE rotation quality checks failed: {quality['checks']}")
            if not task.planned_joints_legal()[0]:raise RuntimeError("Expert trajectory violates URDF limits")
            task.save_dir=str(args.output/"trajectories")
            task.save_traj_data(len(accepted))
            accepted.append({"seed":seed,"native_episode":str(directory),"physical_success":True})
            attempts.append({"seed":seed,"success":True})
        except Exception as error:
            traceback.print_exc();attempts.append({"seed":seed,"success":False,"error":str(error)})
            if interface.task and hasattr(interface.task,"collision_quality"):
                (directory/"collision_quality.json").write_text(json.dumps(interface.task.collision_quality.report(),indent=2)+"\n")
            if interface.recorder:interface.recorder.close(success=False)
        finally:interface.close()
        (args.output/"attempts.json").write_text(json.dumps(attempts,indent=2)+"\n")
    if len(accepted)<args.episodes:raise RuntimeError(f"Only {len(accepted)} of {args.episodes} physical episodes succeeded")
    validation=[]
    for i,item in enumerate(accepted):
        path=args.output/f"w1_finals_real/w1_g2/data/episode_{i:07d}.hdf5"
        convert(Path(item["native_episode"]),path)
        result=validate(path);validation.append(result)
        path.with_suffix(".validation.json").write_text(json.dumps(result,indent=2)+"\n")
        archive=args.output/f"native_episode_{i:04d}.zip"
        with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=3) as file:
            for member in sorted(Path(item["native_episode"]).rglob("*")):
                if member.is_file():file.write(member,Path(f"episode_{i:04d}")/member.relative_to(item["native_episode"]))
        item["native_archive"]=str(archive)
    changed=[name for name,sha in hashes.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha]
    if changed:raise RuntimeError(f"Inputs changed during collection: {changed}")
    report={"config":args.config,"independent_simulation_episodes":len(accepted),"attempts":attempts,
            "accepted":accepted,"validation":validation,"model_training_started":False,"lerobot_exported":False,
            "native_schema":"W1 v0.1 JSONL 250Hz + RGB-D 30Hz + TSF/IDX","canonical_vla":"30Hz / 14D / 3 RGB-D views"}
    (args.output/"collection_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
