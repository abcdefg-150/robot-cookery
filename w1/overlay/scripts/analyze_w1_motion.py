"""Measure whole-path EE rotation in a native W1 demonstration."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from w1_recording import load_actions
from w1_motion_quality import rotation_path_metrics


def analyze(directory):
    directory=Path(directory)
    info=json.loads((directory/'scene_info.json').read_text())
    meta=json.loads((directory/'meta.json').read_text())
    packets={s:load_actions(directory,s) for s in ('left','right')}
    start=meta['created_at']
    transitions=info['fsm_transitions']
    results=[];intervals=[]
    if info.get('paired_foods'):
        for side,timeline in info['arm_fsm_transitions'].items():
            for index,transition in enumerate(timeline):
                if not transition['state'].startswith(('SMALL_PEPPER','POTATO','GREEN_PEPPER','EGGPLANT')):continue
                end=timeline[index+1]['time'] if index+1<len(timeline) else meta['finished_at']-start
                intervals.append((side,transition['state'],transition['time'],end,index))
    for index,transition in enumerate(transitions):
        phase=transition['state'];begin=transition['time']
        end=transitions[index+1]['time'] if index+1<len(transitions) else meta['finished_at']-start
        if phase.startswith(('SMALL_PEPPER','POTATO')):side='right'
        elif phase.startswith(('GREEN_PEPPER','EGGPLANT')):side='left'
        elif phase.startswith(('SPATULA','STIR')):side=info['tool_arm']
        elif phase=='FINAL_PARK':side=info['tool_arm']
        elif phase in ('FINAL_LEFT_PARK','TOOL_CLEARANCE_PARK'):side='left'
        else:continue
        if info.get('paired_foods') and phase.startswith(('SMALL_PEPPER','POTATO','GREEN_PEPPER','EGGPLANT')):continue
        intervals.append((side,phase,begin,end,index))
    for side,phase,begin,end,index in intervals:
        packet=packets[side];times=packet['timestamp']-start
        mask=(times>=begin)&(times<end)
        if np.count_nonzero(mask)<2:continue
        ee=packet['ee_positions'][mask]
        results.append({'phase':phase,'stage_index':index,'arm':side,'duration_s':float(end-begin),
                        'translation_path_m':float(np.linalg.norm(np.diff(ee[:,:3],axis=0),axis=1).sum()),
                        **rotation_path_metrics(Rotation.from_quat(ee[:,3:]).as_matrix())})
    transfers=[r for r in results if r['phase'].endswith('_TRANSFER')]
    entry=[r for r in results if r['phase']=='STIR_APPROACH']
    lower=[r for r in results if r['phase']=='STIR_LOWER']
    checks={'all_four_food_transfers_measured':len(transfers)==4,
            'food_transfer_rotation_below_30deg':len(transfers)==4 and all(r['angular_path_deg']<=30 for r in transfers),
            'stir_entry_below_75deg':len(entry)==1 and entry[0]['angular_path_deg']<=75,
            'stir_lower_keeps_orientation':len(lower)==1 and lower[0]['angular_path_deg']<=15,
            'no_extra_rotation_loops':bool(results) and all(r['excess_rotation_deg']<=25 for r in results)}
    coordination=None
    if info.get('paired_foods'):
        from scipy.ndimage import gaussian_filter1d
        t=packets['left']['timestamp'];dt=float(np.median(np.diff(t)))
        activity=[]
        for side in ('left','right'):
            q=gaussian_filter1d(packets[side]['joint_positions'],2,axis=0)
            activity.append(np.max(abs(np.gradient(q,dt,axis=0)),axis=1)>.03)
        end=max(row['time'] for rows in info['arm_fsm_transitions'].values() for row in rows
                if row['state']=='IDLE' or row['state'].startswith(('SMALL_PEPPER','POTATO','GREEN_PEPPER','EGGPLANT')))
        concurrent=float(np.count_nonzero(activity[0]&activity[1]&((t-start)<end))*dt)
        coordination={'observed_both_arms_moving_s':concurrent,'activity_threshold_rad_s':.03,
                      'planned':info['dual_coordination']}
        from scipy.signal import savgol_filter
        smoothness={}
        for side in ('left','right'):
            packet=packets[side];mask=(packet['timestamp']-start)<end
            points=packet['ee_positions'][mask,:3];joints=packet['joint_positions'][mask]
            residual=points-savgol_filter(points,31,3,axis=0)
            joint_residual=joints-savgol_filter(joints,31,3,axis=0)
            smoothness[side]={'ee_high_frequency_rms_m':float(np.sqrt(np.mean(np.sum(residual**2,axis=1)))),
                              'ee_high_frequency_p99_m':float(np.percentile(np.linalg.norm(residual,axis=1),99)),
                              'joint_high_frequency_rms_rad':float(np.sqrt(np.mean(joint_residual**2)))}
        coordination['actual_smoothness']=smoothness
        coordination['smoothness_method']='250 Hz feedback minus 31-sample cubic Savitzky-Golay trend, paired-food stages only'
        checks['actual_dual_arm_overlap_over_5s']=concurrent>5.
        checks['paired_collision_sweeps_passed']=info['dual_coordination']['minimum_predicted_clearance_m']>=info['dual_coordination']['required_clearance_m']
        checks['dual_ee_high_frequency_rms_below_1mm']=all(s['ee_high_frequency_rms_m']<.001 for s in smoothness.values())
        checks['dual_command_jerk_bounded']=all(p['jerk_rad_s3']<=8.001 for stage in info['dual_coordination']['stages'] for p in stage['command_peaks'].values())
    return {'episode':str(directory),'quaternion_order':'xyzw','sampling_hz':meta['extra']['physics_hz'],
            'metric':'SO(3) geodesic increments over every recorded physical sample',
            'success':all(checks.values()),'checks':checks,'stages':results,'dual_coordination':coordination}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--require-pass',action='store_true')
    args=parser.parse_args();report=analyze(args.episode)
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if args.require_pass and not report['success']:raise SystemExit(1)


if __name__=='__main__':main()
