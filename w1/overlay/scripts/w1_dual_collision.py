"""Paired collision-sphere sweeps in world coordinates, including held foods."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


class SphereKinematics:
    def __init__(self,urdf,spheres):
        root=ET.parse(urdf).getroot();self.joints=[]
        for joint in root.findall('joint'):
            origin=joint.find('origin');transform=np.eye(4)
            if origin is not None:
                transform[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                transform[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            axis=joint.find('axis')
            self.joints.append((joint.get('name'),joint.get('type'),joint.find('parent').get('link'),
                                joint.find('child').get('link'),transform,
                                np.fromstring(axis.get('xyz','1 0 0'),sep=' ') if axis is not None else np.array([1.,0,0])))
        children={j[3] for j in self.joints};self.base=next(l.get('name') for l in root.findall('link') if l.get('name') not in children)
        self.spheres=yaml.safe_load(Path(spheres).read_text())['collision_spheres']

    def forward(self,paths,base):
        count=len(next(iter(paths.values())));poses={self.base:np.broadcast_to(base,(count,4,4)).copy()}
        pending=list(self.joints)
        while pending:
            remaining=[]
            for name,kind,parent,child,origin,axis in pending:
                if parent not in poses:remaining.append((name,kind,parent,child,origin,axis));continue
                local=np.broadcast_to(origin,(count,4,4)).copy()
                values=np.asarray(paths.get(name,np.zeros(count)))
                if kind=='revolute':local[:,:3,:3]=origin[:3,:3]@Rotation.from_rotvec(values[:,None]*axis).as_matrix()
                elif kind=='prismatic':local[:,:3,3]+=values[:,None]*(origin[:3,:3]@axis)
                elif kind!='fixed':raise ValueError(f'Unsupported joint type: {kind}')
                poses[child]=poses[parent]@local
            if len(remaining)==len(pending):raise ValueError('Disconnected URDF')
            pending=remaining
        pieces=[]
        for link,items in self.spheres.items():
            centers=np.array([s['center'] for s in items]);radii=np.array([s['radius'] for s in items])
            points=centers@poses[link][:,:3,:3].transpose(0,2,1)+poses[link][:,None,:3,3]
            pieces.append(np.concatenate([points,np.broadcast_to(radii[None,:,None],(count,len(radii),1))],axis=-1))
        return np.concatenate(pieces,axis=1),poses


def food_spheres(vertices,spacing=.012):
    """A cylinder cover: every point in the convex food lies in a slice sphere."""
    center=np.mean(vertices,axis=0);_,_,axes=np.linalg.svd(vertices-center,full_matrices=False)
    local=(vertices-center)@axes.T;low,high=local[:,0].min(),local[:,0].max()
    n=max(1,int(np.ceil((high-low)/spacing)));width=(high-low)/n
    x=low+(np.arange(n)+.5)*width
    radius=np.sqrt(np.max(np.sum(local[:,1:]**2,axis=1))+(width/2)**2)+.001
    return np.c_[center+x[:,None]*axes[0],np.full(n,radius)]


class DualCollisionChecker:
    def __init__(self,task,margin=.012):
        self.task=task;self.margin=margin
        root=Path(__file__).resolve().parents[1]
        self.model=SphereKinematics(root/'assets/embodiments/w1-g2-black-real-v2/w1_g2.urdf',
                                    root/'assets/embodiments/w1-g2-black-real-v2/collision_spheres.yml')
        self.foods={name:food_spheres(vertices) for name,vertices in task.food_vertices.items()}
        self.kinematics_max_link_error_m=0.
        for side in ('left','right'):
            entity=getattr(task.robot,f'{side}_entity');names=[j.name for j in entity.get_active_joints()];q=entity.get_qpos()
            paths={name:np.asarray([q[i]]) for i,name in enumerate(names)}
            base=getattr(task.robot,f'{side}_entity_origion_pose').to_transformation_matrix()
            _,poses=self.model.forward(paths,base)
            for link in entity.get_links():
                if link.name not in poses:continue
                error=float(np.linalg.norm(poses[link.name][0,:3,3]-link.pose.p))
                self.kinematics_max_link_error_m=max(self.kinematics_max_link_error_m,error)
        if self.kinematics_max_link_error_m>.001:
            raise RuntimeError(f'Dual sphere FK disagrees with physical URDF: {self.kinematics_max_link_error_m:.6f}m')

    def spheres(self,side,positions,open_gripper=False):
        entity=getattr(self.task.robot,f'{side}_entity');names=[j.name for j in entity.get_active_joints()];q=entity.get_qpos()
        paths={f'joint{i+1}':positions[:,i] for i in range(6)}
        for name in ('g2_left_joint','g2_right_joint'):paths[name]=np.full(len(positions),.036 if open_gripper else q[names.index(name)])
        base=getattr(self.task.robot,f'{side}_entity_origion_pose').to_transformation_matrix()
        spheres,poses=self.model.forward(paths,base)
        held=self.task.held_entity.get(side)
        if held is not None and held is not self.task.spatula:
            name=next(k for k,v in self.task.foods.items() if v is held)
            tcp=poses['link6'].copy();tcp[:,:3,3]+=tcp[:,:3,:3]@np.array([0,0,.2466])
            tcp[:,:3,:3]=tcp[:,:3,:3]@Rotation.from_euler('y',-np.pi/2).as_matrix()
            transform=tcp@self.task.held_frames[side];cover=self.foods[name]
            world=cover[:,:3]@transform[:,:3,:3].transpose(0,2,1)+transform[:,None,:3,3]
            spheres=np.concatenate([spheres,np.concatenate([world,np.broadcast_to(cover[None,:,3:],(len(positions),len(cover),1))],axis=-1)],axis=1)
        return spheres

    def clearance(self,left,right,open_gripper=False):
        import torch
        a=self.spheres('left',left,open_gripper);b=self.spheres('right',right,open_gripper)
        minimum=float('inf')
        for begin in range(0,len(a),24):
            x=torch.as_tensor(a[begin:begin+24],dtype=torch.float32,device='cuda')
            y=torch.as_tensor(b[begin:begin+24],dtype=torch.float32,device='cuda')
            gap=torch.cdist(x[:,:,:3],y[:,:,:3])-x[:,:,3,None]-y[:,None,:,3]
            minimum=min(minimum,float(gap.min().item()))
            if minimum<self.margin:return minimum
        return minimum
