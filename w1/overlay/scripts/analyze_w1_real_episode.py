"""Analyze native W1 telemetry, calibrated frames and manually reviewed phases."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import os
import zipfile
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from w1_recording import load_actions, read_tsf, stream_stats, interpolate,MASTER_GRIPPER_STROKE_M,SLAVE_GRIPPER_STROKE_M

ROOT = Path(__file__).resolve().parents[1]
PHASES = [
    (0, 3.5, "READY", "both", "Closed grippers; both arms parked."),
    (3.5, 17, "SMALL_PEPPER_INTO_PAN", "right", "Grasp ~7.3s; release ~13.2s; return then close empty gripper."),
    (17, 33, "GREEN_PEPPER_INTO_PAN", "left", "Grasp ~24.3s; release ~28.1s."),
    (33, 52, "POTATO_INTO_PAN", "right", "Grip ~38.5s at nearly full opening; release ~44-47s."),
    (52, 57, "REPOSITION", "left", "Brief empty gripper test/reposition, not a vegetable transfer."),
    (57, 68, "EGGPLANT_INTO_PAN", "left", "Grasp ~60.8s; release ~64.1s."),
    (68, 78, "SPATULA_PICKUP", "right", "Approach table handle; close ~75.7s."),
    (78, 94, "STIR_CONTENTS", "right", "Repeated scoop strokes; pan rotates; left arm stays parked."),
    (94, 99, "SPATULA_RETURN", "right", "Place tool on table; release ~96.6s."),
    (99, 106.43, "FINISH", "both", "Tool returned; all four props visibly in pan."),
]


def analyze(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".zip":
        directory = ROOT/"data/real_w1"/source.stem
        if not (directory/"meta.json").exists():
            with zipfile.ZipFile(source) as archive:
                for info in archive.infolist():
                    path = Path(info.filename)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError("Unsafe archive member")
                archive.extractall(directory.parent)
    else:
        directory = source
    meta = json.loads((directory/"meta.json").read_text())
    start = float(meta["created_at"])
    report = {"source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,
              "duration_s": meta["finished_at"]-start, "timestamp_dtype": "float64",
              "timestamp_schema_warning": "Original schema says float32; epoch seconds must NOT be cast to float32.",
              "dimensions_user_confirmed": {"base_separation_m": .60, "pan_mouth_diameter_m": .26},
              "gripper_mapping":{"slave_stroke_m":SLAVE_GRIPPER_STROKE_M,"master_stroke_m":MASTER_GRIPPER_STROKE_M,
                  "evidence":"G2 virtual joint limit 72 mm; supplied replay E2 virtual joint limit 47.9 mm; recorded master range about 48.5 mm.",
                  "normalization":"state=slave_width/0.072; action=master_width/0.0479"},
              "arms": {}, "cameras": {}, "phases": [], "real_success_evidence": {
                  "all_four_props_visibly_in_pan": True, "tool_visibly_returned": True,
                  "automated_ground_truth_success": None, "verified_effective_turn_count": None,
                  "reason": "No object poses, contact labels or success annotations are recorded. Two apparent strokes do not prove two contact-verified 45 degree turns."}}
    spec = importlib.util.spec_from_file_location("w1_urdf_ik", ROOT/"envs/robot/ik.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ik = module.URDFPoseIK(ROOT/"assets/embodiments/w1-g2/w1_g2_planner.urdf", "base_link", "link6", [f"joint{i}" for i in range(1,7)])
    streams = {}
    for side in ("left", "right"):
        a = load_actions(directory, side)
        streams[side] = a
        np.savez_compressed(output/f"{side}.npz", **a)
        times = a["timestamp"]
        grid = np.arange(times[0], times[-1], .004)
        q = gaussian_filter1d(interpolate(grid, times, a["joint_positions"]), 3, axis=0)
        p = gaussian_filter1d(interpolate(grid, times, a["ee_positions"][:,:3]), 3, axis=0)
        speed = np.linalg.norm(np.gradient(p, .004, axis=0), axis=1)
        qspeed = np.abs(np.gradient(q, .004, axis=0))
        # Apparent teleoperation lag, not a measured network/control latency.
        reference=a["master_qpos"][:,:6]
        moving=np.linalg.norm(np.gradient(q,.004,axis=0),axis=1)>.15
        candidates=np.arange(-.20,.301,.004)
        scores=np.asarray([np.mean((q[moving]-interpolate(grid[moving]-lag,times,reference))**2) for lag in candidates])
        best=int(scores.argmin())
        ids = np.arange(0, len(times), 200)
        fk = np.asarray([ik.forward(values) for values in a["joint_positions"][ids]])
        logged = a["ee_positions"][ids]
        offsets = np.einsum("nji,nj->ni", fk[:,:3,:3], logged[:,:3]-fk[:,:3,3])
        fixed = Rotation.from_matrix(np.einsum("nji,njk->nik", fk[:,:3,:3], Rotation.from_quat(logged[:,3:]).as_matrix())).mean()
        predicted = fk[:,:3,3]+np.einsum("nij,j->ni", fk[:,:3,:3], [0,0,.2466])
        error = np.linalg.norm(predicted-logged[:,:3], axis=1)
        active = (times-start>70)&(times-start<98)
        grip = gaussian_filter1d(a["gripper"], 12)
        rate = np.gradient(grip, times)
        indices = np.flatnonzero(abs(rate)>.06)
        groups = np.split(indices, np.flatnonzero(np.diff(times[indices])>.2)+1)
        changes = [{"start_s": float(times[g[0]]-start), "end_s": float(times[g[-1]]-start),
                    "direction": "open" if np.median(rate[g])>0 else "close", "width_before_m": float(grip[g[0]]),
                    "width_after_m": float(grip[g[-1]]), "ee_base_xyz": a["ee_positions"][g[len(g)//2],:3].tolist()}
                   for g in groups if len(g)>10]
        report["arms"][side] = {**stream_stats(times), "joint_min_rad": a["joint_positions"].min(0).tolist(),
                "joint_max_rad": a["joint_positions"].max(0).tolist(), "joint_velocity_p99_rad_s": np.percentile(qspeed,99,axis=0).tolist(),
                "ee_speed_p95_m_s": float(np.percentile(speed,95)), "ee_path_length_m": float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),
                "gripper_width_range_m": np.percentile(a["gripper"],[0,100]).tolist(),
                "master_gripper_width_range_m":np.percentile(a["master_qpos"][:,6],[0,100]).tolist(),
                "state_master_joint_error_p95_rad": np.percentile(abs(a["joint_positions"]-a["master_qpos"][:,:6]),95,axis=0).tolist(),
                "apparent_master_to_slave_lag_s": float(candidates[best]),"lag_aligned_joint_rmse_rad":float(np.sqrt(scores[best])),
                "lag_interpretation":"Includes teleop filtering and timestamp/acquisition effects; not sufficient to identify actuator/network latency.",
                "tcp_offset_estimate_m": np.median(offsets,axis=0).tolist(), "tcp_rotation_from_flange": fixed.as_matrix().tolist(),
                "urdf_tcp_position_error_p95_m": float(np.percentile(error,95)),
                "stir_phase_ee_extent_m": np.ptp(a["ee_positions"][active,:3],axis=0).tolist(), "gripper_changes": changes}
    for name in ("cam_front", "cam_left", "cam_right"):
        report["cameras"][name] = {}
        for kind, extension in (("rgb", "mp4"), ("depth", "mkv")):
            file = directory/"sensors"/f"{name}_{kind}.{extension}"
            times = read_tsf(file)
            capture = cv2.VideoCapture(str(file))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            report["cameras"][name][kind] = {**stream_stats(times), "media_frames": frames,
                        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "timestamps_match_frames": len(times)==frames}
            capture.release()
    for begin,end,name,arm,evidence in PHASES:
        report["phases"].append({"start_s": begin,"end_s": end,"phase": name,"active_arm": arm,
                                 "evidence": evidence,"annotation_method": "manual_video_review_plus_joint_and_gripper_telemetry"})
    report["car"] = {"usable": False, "reason": "v=w=-100 are sentinel-like values; exclude from robot motion targets."}
    report["calibration"] = {"front": meta["sensors"]["cam_front"]["calibration"],
        "front_extrinsics_interpretation": "T_arm_base_from_camera_optical; validate on hardware before metric deployment",
        "wrist_intrinsics_and_extrinsics": "not provided; simulation wrist mounts remain an approximation"}
    (output/"analysis.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    (output/"phase_annotations.json").write_text(json.dumps(report["phases"],indent=2)+"\n")
    os.environ.setdefault("MPLCONFIGDIR",str(output/".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(4,1,figsize=(14,11),sharex=True)
    for side,color in (("left","tab:blue"),("right","tab:orange")):
        a=streams[side];t=a["timestamp"]-start
        axes[0].plot(t,a["gripper"]*1000,label=side,color=color)
        axes[1].plot(t,a["ee_positions"][:,2],label=side,color=color)
        step=5;tg=t[::step];p=gaussian_filter1d(a["ee_positions"][::step,:3],2,axis=0)
        axes[2].plot(tg,np.linalg.norm(np.gradient(p,tg,axis=0),axis=1),label=side,color=color)
        axes[3].plot(t,np.linalg.norm(a["effort"][:,:6],axis=1),label=side,color=color)
    for ax,label in zip(axes,("Gripper width (mm)","Reported TCP z in base (m)","TCP speed (m/s)","Joint effort vector norm (logged units)")):
        ax.set_ylabel(label);ax.legend();ax.grid(alpha=.25)
        for begin,_,name,_,_ in PHASES: ax.axvline(begin,color="grey",alpha=.25)
    axes[-1].set_xlabel("Seconds from episode start")
    figure.tight_layout();figure.savefig(output/"trajectory_summary.png",dpi=140);plt.close(figure)
    print(json.dumps({"duration_s":report["duration_s"],"arms":{k:{x:v[x] for x in ("samples","median_hz","ee_speed_p95_m_s","urdf_tcp_position_error_p95_m")} for k,v in report["arms"].items()},"output":str(output)},indent=2))
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"data/w1_real_analysis/episode_0001")
    options=parser.parse_args();analyze(options.source,options.output)
