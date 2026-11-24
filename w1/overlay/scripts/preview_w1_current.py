"""Record the current scene/expert for inspection, without exporting training data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import traceback
import yaml

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from w1_sim_interface import ROOT, W1Simulation


class PreviewRecorder:
    def __init__(self, directory, task, fps=15):
        self.directory = directory
        self.task = task
        self.fps = fps
        self.frames = 0
        self.samples = 0
        self.phase_markers = []
        self.last_phase = None
        self.last_image = None
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
        self.encoder = subprocess.Popen([
            'ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-video_size', '960x536', '-framerate', str(fps), '-i', '-', '-an',
            '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '20',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(directory / 'current_scene_1x.mp4')], stdin=subprocess.PIPE)

    def capture(self, seconds):
        task = self.task
        task._update_render()
        task.cameras.update_picture()
        cameras = [task.cameras.static_camera_list[task.cameras.head_camera_id],
                   task.cameras.left_camera, task.cameras.right_camera]
        pictures = [Image.fromarray(np.uint8(np.clip(c.get_picture('Color')[..., :3] * 255, 0, 255)))
                    for c in cameras]
        sheet = Image.new('RGB', (960, 536), (22, 25, 29))
        sheet.paste(pictures[0], (0, 28))
        sheet.paste(pictures[1].resize((320, 240)), (640, 28))
        sheet.paste(pictures[2].resize((320, 240)), (640, 268))
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 6), f'CURRENT W1 | 1x | t={seconds:6.2f}s | {task.fsm_state}', font=self.font, fill='white')
        rule='Contact + reciprocal stirring >=2s' if task.stir_task else 'Legacy turning check still active'
        draw.text((10, 514), f'PREVIEW ONLY | Front + left/right wrists | {rule}',
                  font=self.font, fill=(255, 212, 120))
        if self.last_phase != task.fsm_state:
            self.phase_markers.append({'phase': task.fsm_state, 'simulation_time_s': float(seconds),
                                       'video_time_s': self.frames / self.fps})
            self.last_phase = task.fsm_state
            pictures[0].save(self.directory / f'stage_{len(self.phase_markers):02d}_{task.fsm_state}.png')
        self.last_image = sheet
        self.encoder.stdin.write(sheet.tobytes())
        self.frames += 1

    def observe(self, task):
        relative = task._record_step * task.scene.get_timestep()
        if relative * self.fps + 1e-8 < self.samples + 1:
            return
        self.capture(relative)
        self.samples += 1
        if self.samples % (self.fps * 20) == 0:
            print(f'[preview] {relative:.1f}s {task.fsm_state}', flush=True)

    def close(self, note):
        if self.last_image is not None:
            image = self.last_image.copy()
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 960, 28), fill=(22, 25, 29))
            draw.text((10, 6), note[:112], font=self.font, fill=(255, 212, 120))
            image.save(self.directory / 'last_frame.png')
            for _ in range(2 * self.fps):
                self.encoder.stdin.write(image.tobytes())
                self.frames += 1
        self.encoder.stdin.close()
        if self.encoder.wait():
            raise RuntimeError('Preview video encoder failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='w1_finals_real_v7')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/w1_current_preview_back_up_20261122')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = ROOT / f'env_cfg/task_config/{args.config}.yml'
    shutil.copy2(config, args.output / 'task_config.yml')
    manifest = ROOT / yaml.safe_load(config.read_text())['asset_manifest']
    metadata = ROOT / yaml.safe_load(manifest.read_text())['spatula']['metadata']
    sources = [config, ROOT / 'envs/w1_finals_real.py', ROOT / 'envs/w1_finals.py',
               ROOT / 'envs/utils/finals_assets.py', manifest, metadata,
               ROOT / 'env_cfg/w1_wrist_image_fit_v1.json']
    report = {'config': args.config, 'seed': args.seed, 'training_data': False,
              'new_2_second_stir_criterion_implemented': False,
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    env = W1Simulation(args.config)
    recorder = None
    try:
        print('Initializing current simulation for preview', flush=True)
        env.reset(args.seed)
        task = env.task
        report['new_2_second_stir_criterion_implemented']=task.stir_task
        task.policy_observer = None
        work_normal = task.spatula.pose.to_transformation_matrix()[:3, :3] @ np.array([0, 0, 1])
        report.update(spatula_initial_surface=task.tool_metadata.get('initial_surface', 'working_face_up'),
                      settled_working_face_normal_world=work_normal.tolist())
        if task.tool_metadata.get('initial_surface') == 'back_up' and work_normal[2] > -.8:
            raise RuntimeError('Spatula did not settle with its back facing upward')
        recorder = PreviewRecorder(args.output, task)
        recorder.capture(0)
        for _ in range(29):
            recorder.encoder.stdin.write(recorder.last_image.tobytes())
            recorder.frames += 1
        task.record_sink = recorder
        try:
            info, success = env.expert()
            report.update(task_success=bool(success), scene_info=info, stop_reason='expert_finished')
        except Exception as error:
            traceback.print_exc()
            report.update(task_success=False, stop_reason=str(error), last_phase=task.fsm_state)
        finally:
            task.record_sink = None
            recorder.capture(task._record_step * task.scene.get_timestep())
            report.update(collision_quality=task.collision_quality.report(),
                          simulation_duration_s=task._record_step * task.scene.get_timestep(),
                          phase_markers=recorder.phase_markers,
                          turn_events=task.flip_events)
            note = ('Preview complete; training acceptance not evaluated' if report.get('task_success') else
                    'Preview stopped by current expert checks. See preview_report.json.')
            recorder.close(note)
            report.update(video_frames=recorder.frames, video_fps=recorder.fps)
    finally:
        env.close()
        (args.output / 'preview_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('config', 'task_success', 'stop_reason', 'simulation_duration_s')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
