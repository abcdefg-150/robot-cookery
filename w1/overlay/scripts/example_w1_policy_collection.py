"""Smoke-test the VLA-facing interface with a safe hold policy in simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
from w1_sim_interface import W1Simulation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="w1_finals_real_v5")
    parser.add_argument("--steps",type=int,default=30)
    parser.add_argument("--record-dir",type=Path)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args();env=W1Simulation(args.config)
    try:
        observation=env.reset(seed=0,record_dir=args.record_dir)
        initial=observation["state"].copy();count=0
        for _ in range(args.steps):
            # Replace this hold action with your VLA's absolute joint targets.
            # Gripper components 6/13 are total opening widths in metres.
            observation,reward,terminated,truncated,info=env.step(initial);count+=info["physics_steps"]
            assert not terminated and reward==0 and not truncated,"A hold policy must not satisfy the task"
            assert np.isfinite(observation["state"]).all()
        assert count==round(args.steps*250/30)
        assert all(x.shape==(480,640,3) and x.dtype==np.uint8 for x in observation["images"].values())
        assert all(x.shape==(480,640) and x.dtype==np.uint16 and np.any(x>0) for x in observation["depths"].values())
        result={"control_steps":args.steps,"physics_steps":count,"state_shape":list(observation["state"].shape),
                "cameras":{k:list(v.shape) for k,v in observation["images"].items()},
                "depth_unit":observation["depth_unit"],"task_success":info["success"],
                "policy":"hold","training_demonstration":False,"status":"passed"}
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
    finally:env.close()


if __name__=="__main__":main()
