"""Measure small reciprocal stirring from physical motion and real contacts."""
import numpy as np


class StirMonitor:
    def __init__(self, minimum_motion_s=2.0, minimum_food_motion_m=.005, stroke_m=.008):
        self.minimum_motion_s=float(minimum_motion_s)
        self.minimum_food_motion_m=float(minimum_food_motion_m)
        self.stroke_m=float(stroke_m)
        self.reset()

    def reset(self):
        self.active=False;self.motion_s=0.;self.contact_s=0.;self.contact_steps=0
        self.reversals=0;self.peak_food_motion=0.;self.contacted=set()
        self.previous=None;self.anchor=None;self.extreme=None;self.direction=0
        self.axis=None;self.initial_foods={};self.path_m=0.
        self.time_since_contact=float('inf')

    def begin(self, axis, foods):
        self.reset();self.active=True
        axis=np.asarray(axis,dtype=float)[:2];self.axis=axis/np.linalg.norm(axis)
        self.initial_foods={k:np.asarray(v,dtype=float).copy() for k,v in foods.items()}

    def observe(self, dt, tool_xy, inside_pan, contacted_foods, foods):
        if not self.active:return
        point=np.asarray(tool_xy,dtype=float)[:2]
        if self.previous is None:self.previous=point.copy()
        distance=float(np.linalg.norm(point-self.previous));self.previous=point.copy()
        if not inside_pan:
            self.anchor=None;self.extreme=None;self.direction=0
            return
        contacts=set(contacted_foods)&set(self.initial_foods)
        self.time_since_contact+=dt
        if contacts:
            self.contacted.update(contacts);self.contact_s+=dt;self.contact_steps+=1
            self.time_since_contact=0.
        # Count actual in-pan motion after contact, never stationary waiting.
        if self.contacted and self.time_since_contact<=.5 and distance/dt>.003:
            self.motion_s+=dt;self.path_m+=distance
        for name in self.contacted:
            self.peak_food_motion=max(self.peak_food_motion,float(np.linalg.norm(np.asarray(foods[name])-self.initial_foods[name])))
        coordinate=float(point@self.axis)
        if self.anchor is None:self.anchor=self.extreme=coordinate
        if self.direction==0:
            if abs(coordinate-self.anchor)>=self.stroke_m:
                self.direction=1 if coordinate>self.anchor else -1;self.extreme=coordinate
        elif self.direction*(coordinate-self.extreme)>=0:self.extreme=coordinate
        elif self.direction*(coordinate-self.extreme)<=-self.stroke_m:
            self.reversals+=1;self.direction*=-1;self.extreme=coordinate

    @property
    def success(self):
        return bool(self.motion_s+1e-9>=self.minimum_motion_s and self.contact_steps>=2
                    and self.reversals>=1 and self.peak_food_motion>=self.minimum_food_motion_m)

    def report(self):
        return {'success':self.success,'motion_duration_s':self.motion_s,'contact_duration_s':self.contact_s,
                'contact_steps':self.contact_steps,'direction_reversals':self.reversals,
                'maximum_food_displacement_m':self.peak_food_motion,'contacted_foods':sorted(self.contacted),
                'tool_planar_path_m':self.path_m,'minimum_motion_duration_s':self.minimum_motion_s,
                'minimum_food_displacement_m':self.minimum_food_motion_m,
                'minimum_stroke_m':self.stroke_m,
                'duration_definition':'Measured in-pan tool motion after first food contact; stationary waits excluded.'}
