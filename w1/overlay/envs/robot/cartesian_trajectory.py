"""Continuous URDF IK and bounded, time-scaled joint trajectories.

The caller must collision-check every returned control sample before execution.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp


def time_parameterize(knots, velocity_limits, acceleration_limits, dt=.004):
    knots=np.asarray(knots,dtype=float)
    if len(knots)<2:raise ValueError("At least two joint knots are required")
    curve=PchipInterpolator(np.linspace(0,1,len(knots)),knots,axis=0)
    u=np.linspace(0,1,2001)
    phase=10*u**3-15*u**4+6*u**5
    speed=30*u**2-60*u**3+30*u**4
    accel=60*u-180*u**2+120*u**3
    dq=curve.derivative()(phase)*speed[:,None]
    ddq=curve.derivative(2)(phase)*speed[:,None]**2+curve.derivative()(phase)*accel[:,None]
    duration=1.04*max(.16,float(np.max(abs(dq)/np.asarray(velocity_limits))),
                     float(np.sqrt(np.max(abs(ddq)/np.asarray(acceleration_limits)))))
    count=max(2,int(np.ceil(duration/dt))+1);duration=(count-1)*dt
    u=np.linspace(0,1,count);phase=np.clip(10*u**3-15*u**4+6*u**5,0,1)
    speed=(30*u**2-60*u**3+30*u**4)/duration
    positions=curve(phase);velocities=curve.derivative()(phase)*speed[:,None]
    velocities[[0,-1]]=0
    return positions,velocities,duration


def cartesian_knots(ik,current,position,quaternion_wxyz,position_step=.008,angle_step=.05):
    initial=ik.forward(current);rotation=Rotation.from_quat(np.asarray(quaternion_wxyz)[[1,2,3,0]])
    origin_rotation=Rotation.from_matrix(initial[:3,:3])
    angle=(origin_rotation.inv()*rotation).magnitude()
    count=max(2,int(np.ceil(np.linalg.norm(np.asarray(position)-initial[:3,3])/position_step)),int(np.ceil(angle/angle_step)))
    orientations=Slerp([0,1],Rotation.from_matrix(np.stack([initial[:3,:3],rotation.as_matrix()])))
    knots=[np.asarray(current,dtype=float)]
    for alpha in np.linspace(0,1,count+1)[1:]:
        point=(1-alpha)*initial[:3,3]+alpha*np.asarray(position)
        quat=orientations(alpha).as_quat()[[3,0,1,2]]
        solution=ik.solve(point,quat,knots[-1],max_restarts=0)
        if solution is None or np.max(abs(solution-knots[-1]))>.35:return None
        knots.append(solution)
    return np.asarray(knots)
