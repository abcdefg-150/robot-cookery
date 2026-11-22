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
    results=[]
    for index,transition in enumerate(transitions):
        phase=transition['state'];begin=transition['time']
        end=transitions[index+1]['time'] if index+1<len(transitions) else meta['finished_at']-start
        if phase.startswith(('SMALL_PEPPER','POTATO')):side='right'
        elif phase.startswith(('GREEN_PEPPER','EGGPLANT')):side='left'
        elif phase.startswith(('SPATULA','STIR')):side=info['tool_arm']
        elif phase=='FINAL_PARK':side=info['tool_arm']
        elif phase in ('FINAL_LEFT_PARK','TOOL_CLEARANCE_PARK'):side='left'
        else:continue
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
    return {'episode':str(directory),'quaternion_order':'xyzw','sampling_hz':meta['extra']['physics_hz'],
            'metric':'SO(3) geodesic increments over every recorded physical sample',
            'success':all(checks.values()),'checks':checks,'stages':results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--require-pass',action='store_true')
    args=parser.parse_args();report=analyze(args.episode)
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if args.require_pass and not report['success']:raise SystemExit(1)


if __name__=='__main__':main()
