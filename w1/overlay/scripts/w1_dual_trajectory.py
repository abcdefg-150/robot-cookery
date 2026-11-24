"""C2 joint paths, bounded jerk, and stretch-only dual-arm synchronization."""
import numpy as np
from scipy.interpolate import CubicSpline,BPoly


def smooth_trajectory(knots, velocity_limits, acceleration_limits, jerk_limits=8., dt=.004):
    knots=np.asarray(knots,dtype=float)
    if len(knots)<2:raise ValueError('At least two knots required')
    curve=CubicSpline(np.linspace(0,1,len(knots)),knots,axis=0,bc_type='natural')
    def evaluate(u):
        phase=np.clip(10*u**3-15*u**4+6*u**5,0,1)
        d=30*u**2-60*u**3+30*u**4;dd=60*u-180*u**2+120*u**3;ddd=60-360*u+360*u**2
        a=curve(phase,1);b=curve(phase,2);c=curve(phase,3)
        return curve(phase),a*d[:,None],b*d[:,None]**2+a*dd[:,None],c*d[:,None]**3+3*b*d[:,None]*dd[:,None]+a*ddd[:,None]
    _,v,a,j=evaluate(np.linspace(0,1,4001))
    duration=1.08*max(.2,float(np.max(abs(v)/velocity_limits)),
                      float(np.sqrt(np.max(abs(a)/acceleration_limits))),
                      float(np.cbrt(np.max(abs(j)/jerk_limits))))
    count=int(np.ceil(duration/dt))+1;duration=(count-1)*dt
    p,v,a,j=evaluate(np.linspace(0,1,count));v/=duration;a/=duration**2;j/=duration**3
    v[[0,-1]]=0;a[[0,-1]]=0
    return {'status':'Success','position':p,'velocity':v,'acceleration':a,'jerk':j,
            'duration_s':duration,'planner_status':'c2_minimum_jerk_timing','all_control_samples_collision_checked':False}


def stretch_trajectory(result,duration,dt=.004):
    original=result['duration_s']
    if duration+1e-9<original:raise ValueError('Synchronization must not speed up an arm')
    count=int(round(duration/dt))+1
    # Quintic Hermite interpolation retains position, velocity AND
    # acceleration at every original sample, including the quiet endpoints.
    values=[[p,v,a] for p,v,a in zip(result['position'],result['velocity'],result['acceleration'])]
    curve=BPoly.from_derivatives(np.linspace(0,original,len(values)),values)
    factor=original/duration;t=np.linspace(0,original,count)
    output=dict(result,position=curve(t),velocity=curve(t,1)*factor,
                acceleration=curve(t,2)*factor**2,jerk=curve(t,3)*factor**3,duration_s=duration)
    output['velocity'][[0,-1]]=0;output['acceleration'][[0,-1]]=0
    return output
