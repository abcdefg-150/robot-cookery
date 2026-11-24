"""Align real/sim W1 native episodes to a common 30 Hz, 14-D RGB-D HDF5.

No camera FPS relabeling: use each TSF clock, interpolate joint telemetry and
keep the selected original RGB/depth timestamps and indices in the output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import cv2
import h5py
import numpy as np
from w1_recording import CAMERAS,load_actions,read_tsf,nearest_indices,interpolate,SLAVE_GRIPPER_STROKE_M,MASTER_GRIPPER_STROKE_M

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from XPolicyLab.utils.process_data import encode_image_bit,decode_image_bit


def selected_rgb(path,indices):
    capture=cv2.VideoCapture(str(path));current=-1;previous=None
    try:
        for index in indices:
            while current<index:
                ok,frame=capture.read();current+=1
                if not ok:raise ValueError(f"Missing RGB frame {index}: {path}")
                previous=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            yield previous
    finally:capture.release()


def selected_depth(path,indices,width,height):
    command=["ffmpeg","-v","error","-i",str(path),"-map","0:v:0","-vsync","0","-f","rawvideo","-pix_fmt","gray16le","-"]
    process=subprocess.Popen(command,stdout=subprocess.PIPE)
    size=width*height*2;current=-1;previous=None
    try:
        for index in indices:
            while current<index:
                data=process.stdout.read(size);current+=1
                if len(data)!=size:raise ValueError(f"Missing depth frame {index}: {path}")
                previous=np.frombuffer(data,dtype="<u2").reshape(height,width)
            yield previous
        # Drain remaining frames so ffmpeg closes its pipe without SIGPIPE.
        while process.stdout.read(size):pass
        if process.wait():raise RuntimeError(f"Depth decoder failed: {path}")
    finally:
        process.stdout.close()
        if process.poll() is None:process.terminate();process.wait()


def convert(source,output,fps=30):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError(output)
    meta=json.loads((source/"meta.json").read_text());extra=meta.get("extra",{});simulation=extra.get("source")=="simulation"
    stirring=extra.get('task_judgement_version')=='stir_contact_motion_v1'
    slave_stroke=float(extra.get("maximum_width_m",SLAVE_GRIPPER_STROKE_M))
    master_stroke=float(extra.get("master_maximum_width_m",MASTER_GRIPPER_STROKE_M))
    if simulation and not extra.get("success"):raise ValueError("Refusing to export an unsuccessful simulation as training data")
    arms={s:load_actions(source,s) for s in ("left","right")}
    clocks={}
    for name in CAMERAS:
        for kind,ext in (("rgb","mp4"),("depth","mkv")):
            clocks[(name,kind)]=read_tsf(source/f"sensors/{name}_{kind}.{ext}")
    start=max([a["timestamp"][0] for a in arms.values()]+[v[0] for v in clocks.values()])
    end=min([a["timestamp"][-1]-1/fps for a in arms.values()]+[v[-1] for v in clocks.values()])
    epoch=float(meta["created_at"]);first=np.ceil((start-epoch)*fps)/fps
    relative=np.arange(first,(end-epoch)+1e-8,1/fps,dtype=np.float64);times=relative+epoch
    if len(times)<2:raise ValueError("No common camera/action interval")
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_suffix(".hdf5.tmp")
    report={"source":str(source),"source_type":"simulation" if simulation else "real","frames":len(times),"frequency_hz":fps,
        "state_action_dim":14,"duration_s":len(times)/fps,"action_semantics":"next_sample_joint_reference",
        "action_reference_source":"simulated_drive_target" if simulation else "recorded_master_qpos",
        "slave_gripper_stroke_m":slave_stroke,"master_gripper_stroke_m":master_stroke,
        "gripper_mapping":"state=slave_width/0.072; action=master_width/0.0479; raw master and slave widths are different mechanisms",
        "action_source_limitation":None if simulation else "Master encoder reference is recorded; actual slave network command and command latency are not provided.",
        "camera_alignment":{},"physical_success":bool(extra["success"]) if simulation else None,
        "real_turn_count_ground_truth":None if not simulation else len(json.loads((source/"scene_info.json").read_text())["flip_events"])}
    with h5py.File(temporary,"w") as target:
        target.attrs.update(source_type=report["source_type"],action_semantics=report["action_semantics"],
            action_reference_source=report["action_reference_source"],gripper_normalization="state=slave_width/0.072; action=master_width/0.0479",
            source_episode=str(source.resolve()),timestamp_dtype="float64",depth_unit="mm")
        for side,a in arms.items():
            state=interpolate(times,a["timestamp"],a["joint_positions"])
            opening=np.clip(interpolate(times,a["timestamp"],a["gripper"])/slave_stroke,0,1)
            reference=interpolate(times+1/fps,a["timestamp"],a["master_qpos"])
            target.create_dataset(f"state/{side}_arm_joint_states",data=state.astype(np.float32))
            target.create_dataset(f"state/{side}_ee_joint_states",data=opening[:,None].astype(np.float32))
            target.create_dataset(f"action/{side}_arm_joint_states",data=reference[:,:6].astype(np.float32))
            target.create_dataset(f"action/{side}_ee_joint_states",data=np.clip(reference[:,6:7]/master_stroke,0,1).astype(np.float32))
        target.create_dataset("additional_info/frequency",data=float(fps))
        target.create_dataset("additional_info/timestamps",data=relative)
        target.create_dataset("additional_info/action_timestamps",data=relative+1/fps)
        target.create_dataset("additional_info/epoch_start",data=epoch)
        phases=np.full(len(times),"UNANNOTATED",dtype=object)
        if simulation:
            target.create_dataset("metadata/asset_manifest",data=json.dumps(extra["asset_manifest"]),dtype=h5py.string_dtype())
            target.create_dataset("metadata/spatula_metadata",data=json.dumps(extra["spatula_metadata"]),dtype=h5py.string_dtype())
            target.create_dataset("metadata/scene_info",data=(source/"scene_info.json").read_text(),dtype=h5py.string_dtype())
            truth=[json.loads(line) for line in (source/"extra/ground_truth.jsonl").read_text().splitlines()]
            gt_times=np.asarray([r["timestamp"] for r in truth]);ids=nearest_indices(gt_times,times)
            phases=np.asarray([truth[i]["fsm_state"] for i in ids],dtype=object)
            if 'arm_fsm_state' in truth[0]:
                for side in ('left','right'):
                    target.create_dataset(f'additional_info/{side}_fsm_state',data=np.asarray([truth[i]['arm_fsm_state'][side] for i in ids],dtype=object),dtype=h5py.string_dtype())
            for name in truth[0]["poses"]:
                target.create_dataset(f"task/{name}",data=np.asarray([truth[i]["poses"][name] for i in ids]))
            for key in ("effective_flip_count","turn_angle_deg","tool_pepper_contact","tool_returned"):
                target.create_dataset(f"task/{key}",data=np.asarray([truth[i][key] for i in ids]))
            for key in ('stir_success','stir_motion_duration_s','stir_contact_duration_s','stir_reversals','stir_max_food_displacement_m'):
                if key in truth[0]:target.create_dataset(f'task/{key}',data=np.asarray([truth[i][key] for i in ids]))
            target.create_dataset("task/contact_pairs",data=np.asarray([json.dumps(truth[i].get("contacts",[])) for i in ids],dtype=object),dtype=h5py.string_dtype())
            target.create_dataset("task/source_timestamps",data=gt_times[ids]-epoch)
        else:
            annotation=ROOT/"data/w1_real_analysis/episode_0001/phase_annotations.json"
            reference=(ROOT/"data/real_w1/episode_0001").resolve()
            if annotation.exists() and source.resolve()==reference:
                for row in json.loads(annotation.read_text()):phases[(relative>=row["start_s"])&(relative<row["end_s"])]=row["phase"]
        target.create_dataset("additional_info/fsm_state",data=phases,dtype=h5py.string_dtype())
        instruction=("Put both green peppers, the potato and the eggplant in the pan, stir with small back-and-forth pushes for at least two seconds, and return the spatula."
                     if stirring or not simulation else "Put both green peppers, the potato and the eggplant in the pan, turn the contents twice with the spatula, and return the spatula.")
        if simulation and extra.get('paired_foods'):instruction=meta['task_meta']['prompt']
        target.create_dataset("instructions",data=json.dumps([instruction]),dtype=h5py.string_dtype())
        for native,canonical in CAMERAS.items():
            cfg=meta["sensors"][native];width,height=cfg["camera"][0]["width"],cfg["camera"][0]["height"]
            group=target.create_group(f"vision/{canonical}");group.create_dataset("shape",data=[height,width,3])
            calib=cfg.get("calibration",{})
            if "intrinsics" in calib:group.create_dataset("intrinsic_matrix",data=calib["intrinsics"]["camera"])
            for kind,ext in (("rgb","mp4"),("depth","mkv")):
                clock=clocks[(native,kind)];ids=nearest_indices(clock,times);errors=clock[ids]-times
                if abs(errors).max()>1.5/fps:raise ValueError(f"Camera gap is too large: {native}/{kind}")
                group.create_dataset(f"{kind}_source_timestamps",data=clock[ids]-epoch)
                group.create_dataset(f"{kind}_source_frame_indices",data=ids)
                report["camera_alignment"][f"{native}/{kind}"]={"max_error_s":float(abs(errors).max()),
                    "p95_error_s":float(np.percentile(abs(errors),95)),"source_frames":len(clock),"unique_selected_frames":len(np.unique(ids))}
                file=source/f"sensors/{native}_{kind}.{ext}"
                if kind=="rgb":
                    ds=group.create_dataset("colors",shape=(len(times),),dtype=h5py.vlen_dtype(np.uint8))
                    for i,frame in enumerate(selected_rgb(file,ids)):ds[i]=np.frombuffer(encode_image_bit(frame),dtype=np.uint8)
                else:
                    ds=group.create_dataset("depths",shape=(len(times),height,width),dtype=np.uint16,chunks=(1,height,width),compression="lzf")
                    for i,frame in enumerate(selected_depth(file,ids,width,height)):ds[i]=frame
                print(f"[W1 align] {native}/{kind}: {len(times)} frames, max clock difference {abs(errors).max()*1000:.2f} ms",flush=True)
        target.attrs["alignment_report"]=json.dumps(report)
    temporary.replace(output)
    report_path=output.with_suffix(".alignment.json");report_path.write_text(json.dumps(report,indent=2)+"\n")
    return report


def validate(path):
    import xml.etree.ElementTree as ET
    root=ET.parse(ROOT/"w1_urdf/urdf/play_g2.urdf").getroot()
    limits=np.array([[float(root.find(f"joint[@name='joint{i}']/limit").get(x)) for x in ("lower","upper")] for i in range(1,7)])
    with h5py.File(path,"r") as data:
        times=data["additional_info/timestamps"][:];fps=float(data["additional_info/frequency"][()]);n=len(times)
        np.testing.assert_allclose(np.diff(times),1/fps,atol=1e-6)
        np.testing.assert_allclose(data["additional_info/action_timestamps"][:]-times,1/fps,atol=1e-6)
        for group in ("state","action"):
            for side in ("left","right"):
                q=data[f"{group}/{side}_arm_joint_states"][:];g=data[f"{group}/{side}_ee_joint_states"][:]
                if q.shape!=(n,6) or g.shape!=(n,1) or not np.isfinite(q).all() or not np.isfinite(g).all():raise ValueError("Invalid state/action")
                if np.any(q<limits[:,0]-.02) or np.any(q>limits[:,1]+.02) or np.any(g<0) or np.any(g>1):raise ValueError("Joint/width limit violation")
        for camera in CAMERAS.values():
            g=data[f"vision/{camera}"]
            if len(g["colors"])!=n or len(g["depths"])!=n:raise ValueError("Camera length mismatch")
            for i in (0,n//2,n-1):
                if decode_image_bit(g["colors"][i]).shape!=(480,640,3):raise ValueError("Invalid RGB shape")
                if g["depths"][i].shape!=(480,640) or g["depths"].dtype!=np.uint16:raise ValueError("Invalid metric depth")
        if data.attrs["source_type"]=="simulation":
            source=Path(data.attrs["source_episode"]);info=json.loads((source/"scene_info.json").read_text())
            if not all(info["success_checks"].values()):raise ValueError("Physical task did not pass")
            if info.get('task_judgement_version')=='stir_contact_motion_v1':
                metrics=info['stir_metrics']
                if not bool(data['task/stir_success'][-1]) or metrics['motion_duration_s']<2 or metrics['direction_reversals']<1 or metrics['maximum_food_displacement_m']<.005:raise ValueError('Missing successful reciprocal stirring observations')
            elif data["task/effective_flip_count"][-1]!=2:raise ValueError("Missing successful turn observations")
            if not data["task/tool_returned"][-1]:raise ValueError('Missing tool return')
            import trimesh
            from scipy.spatial.transform import Rotation
            assets=json.loads(data["metadata/asset_manifest"].asstr()[()]);pan=info["pan_geometry"]
            for kind in ("small_pepper","green_pepper","potato","eggplant"):
                cfg=assets[kind];mesh=trimesh.load(ROOT/cfg["visual"],force="mesh")
                collider=trimesh.load(ROOT/cfg["collision"],force="mesh")
                center=collider.bounds.mean(0) if cfg["center_visual"] else np.zeros(3)
                local=(mesh.convex_hull.vertices-center)*cfg["scale"]@Rotation.from_euler("xyz",cfg["rotation_rpy"]).as_matrix().T
                poses=data[f"task/{kind}_pose"][:];pose=poses[-1]
                points=local@Rotation.from_quat(pose[[4,5,6,3]]).as_matrix().T+pose[:3]
                if np.max(np.linalg.norm(points[:,:2]-np.asarray(pan["center"])[:2],axis=1))>pan["inner_radius"]+.003:raise ValueError(f"Final {kind} geometry outside pan")
                if not pan["floor"]-.004<points[:,2].min()<pan["rim"]+.006:raise ValueError(f"Final {kind} vertical placement invalid")
                if np.max(np.linalg.norm(poses[-15:,:3]-pose[:3],axis=1))>.005:raise ValueError(f"Final {kind} moving")
        return {"path":str(path),"frames":n,"frequency_hz":fps,"state_action_dim":14,"cameras":3,"rgb_shape":[480,640,3],
                "depth_unit":"mm","status":"passed","source_type":str(data.attrs["source_type"])}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source",type=Path);p.add_argument("output",type=Path);p.add_argument("--fps",type=float,default=30)
    a=p.parse_args();convert(a.source,a.output,a.fps);v=validate(a.output);a.output.with_suffix(".validation.json").write_text(json.dumps(v,indent=2)+"\n");print(json.dumps(v,indent=2))
