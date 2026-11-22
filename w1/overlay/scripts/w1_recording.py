"""W1 v0.1 recording IO. All absolute timestamps stay float64."""
import json
import struct
from pathlib import Path
import numpy as np

CAMERAS = {"cam_front": "cam_head", "cam_left": "cam_left_wrist", "cam_right": "cam_right_wrist"}
FIELDS = ("joint_positions", "ee_positions", "gripper", "effort", "master_qpos", "master_ee_positions", "timestamp")
SLAVE_GRIPPER_STROKE_M = .072
MASTER_GRIPPER_STROKE_M = .0479


def load_actions(directory, side):
    rows = [json.loads(line) for line in (Path(directory)/"actions"/f"{side}_arm.jsonl").read_text().splitlines() if line.strip()]
    result = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in FIELDS}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"Non-finite action data: {side}")
    if np.any(np.diff(result["timestamp"]) <= 0):
        raise ValueError(f"Non-monotonic action timestamps: {side}")
    return result


def read_tsf(video):
    video = Path(video)
    data = video.with_suffix(video.suffix+".tsf").read_bytes()
    if data[:8] != b"TSF1Q   " or (len(data)-16)%8:
        raise ValueError(f"Unknown timestamp format: {video}")
    size = struct.unpack("<Q", data[8:16])[0]
    if size != video.stat().st_size:
        raise ValueError(f"Timestamp file belongs to another media file: {video}")
    times = np.frombuffer(data[16:], dtype="<u8").astype(np.float64)/1000
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"Non-monotonic camera timestamps: {video}")
    return times


def interpolate(times, source_times, values):
    values = np.asarray(values)
    if values.ndim == 1:
        return np.interp(times, source_times, values)
    return np.stack([np.interp(times, source_times, values[:, i]) for i in range(values.shape[1])], axis=-1)


def nearest_indices(source, target):
    right = np.searchsorted(source, target).clip(1, len(source)-1)
    left = right-1
    return np.where(abs(source[left]-target) <= abs(source[right]-target), left, right)


def stream_stats(times):
    delta = np.diff(times)
    return {"samples": len(times), "start_s": float(times[0]), "end_s": float(times[-1]),
            "duration_s": float(times[-1]-times[0]), "median_hz": float(1/np.median(delta)),
            "mean_hz": float((len(times)-1)/(times[-1]-times[0])), "max_gap_s": float(delta.max()),
            "gap_count_over_two_periods": int(np.sum(delta > 2*np.median(delta)))}


class IndexedJSONL:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("wb")
        self.offsets = []

    def write(self, row):
        self.offsets.append(self.file.tell())
        self.file.write((json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n").encode())

    def close(self):
        size = self.file.tell()
        self.file.close()
        self.path.with_suffix(self.path.suffix+".idx").write_bytes(b"IDV1Q   "+struct.pack("<Q", size)+np.asarray(self.offsets, dtype="<u8").tobytes())


def write_tsf(video, times):
    video = Path(video)
    millis = np.rint(np.asarray(times, dtype=np.float64)*1000).astype("<u8")
    video.with_suffix(video.suffix+".tsf").write_bytes(b"TSF1Q   "+struct.pack("<Q", video.stat().st_size)+millis.tobytes())
