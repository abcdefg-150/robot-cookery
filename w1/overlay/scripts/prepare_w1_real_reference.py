"""Create a separate black, 60 cm, 26 cm real-reference scene profile."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
import yaml
from w1_recording import load_actions

ROOT=Path(__file__).resolve().parents[1]


def prepare_pan_collision(assets,folder):
    """Fit the inside surface; dense AI-mesh edges excite rolling contacts."""
    pan=assets["pan"];n=64;floor=pan["floor_height"];rim=pan["rim_height"]
    radii=[.065,.085,.105,.125,.130]
    heights=[floor,.01306732,.03134803,.06582945,rim]
    vertices=[[0,0,floor]];faces=[]
    for radius,height in zip(radii,heights):
        for angle in np.arange(n)*2*np.pi/n:vertices.append([radius*np.cos(angle),radius*np.sin(angle),height])
    for j in range(n):faces.append([0,1+j,1+(j+1)%n])
    for i in range(len(radii)-1):
        a=1+i*n;b=a+n
        for j in range(n):
            k=(j+1)%n;faces.extend([[a+j,b+j,b+k],[a+j,b+k,a+k]])
    inner=1+(len(radii)-1)*n;top=len(vertices)
    for angle in np.arange(n)*2*np.pi/n:vertices.append([.132*np.cos(angle),.132*np.sin(angle),rim])
    bottom=len(vertices)
    for angle in np.arange(n)*2*np.pi/n:vertices.append([.132*np.cos(angle),.132*np.sin(angle),0])
    center=len(vertices);vertices.append([0,0,0])
    for j in range(n):
        k=(j+1)%n
        faces.extend([[inner+j,top+j,top+k],[inner+j,top+k,inner+k],
                      [top+j,bottom+j,bottom+k],[top+j,bottom+k,top+k],[center,bottom+k,bottom+j]])
    mesh=trimesh.Trimesh(np.asarray(vertices)/pan["scale"],np.asarray(faces),process=True)
    mesh.fix_normals()
    if not mesh.is_watertight or mesh.volume<=0:raise ValueError("Invalid fitted pan collision mesh")
    path=folder/"pan_collision.obj";mesh.export(path)
    pan["collision"]=str(path.relative_to(ROOT))
    pan["collision_approximation"]={"method":"closed axisymmetric piecewise-linear inner-surface fit",
        "angular_segments":n,"triangles":len(mesh.faces),"radii_m":radii,"heights_m":heights,
        "visual_mesh_modified":False,"handle_collision_included":False,
        "note":"Fixed pan profile; handle is visual only. Retains bowl diameter and rim height."}


def prepare(episode):
    old=ROOT/"assets/embodiments/w1-g2";new=ROOT/"assets/embodiments/w1-g2-black"
    new.mkdir(parents=True,exist_ok=True)
    for source in old.iterdir():
        if source.is_file():shutil.copy2(source,new/source.name)
    for path in new.glob("*.yml"):
        path.write_text(path.read_text().replace("/w1-g2/","/w1-g2-black/"))
    for name in ("curobo.yml","curobo_tmp.yml"):
        path=new/name;planning=yaml.safe_load(path.read_text());planning["planner"]["reuse_sequential_motion_gen"]=True
        path.write_text(yaml.safe_dump(planning,sort_keys=False))
    for path in new.glob("*.urdf"):
        tree=ET.parse(path)
        for link in tree.getroot().findall("link"):
            for visual in link.findall("visual"):
                material=visual.find("material")
                if material is None:material=ET.SubElement(visual,"material",name="w1_black")
                color=material.find("color")
                if color is None:color=ET.SubElement(material,"color")
                color.set("rgba", "0.035 0.04 0.045 1" if not link.get("name","").startswith("g2_") else "0.35 0.37 0.38 1")
        ET.indent(tree,space="  ");tree.write(path,encoding="utf-8",xml_declaration=True)
    # The real folded home brings link1 close to the wrist. Long slab spheres
    # previously overlap by 21 mm although the CAD bodies are clear. Refine
    # those three meshes rather than disabling their self-collision checks.
    sphere_path=new/"collision_spheres.yml"
    spheres=yaml.safe_load(sphere_path.read_text())
    for name in ("link1","link5"):
        mesh=trimesh.load(ROOT/f"w1_urdf/meshes/{name}.STL",force="mesh")
        low,high=mesh.bounds;counts=np.maximum(1,np.ceil((high-low)/.016).astype(int));width=(high-low)/counts
        radius=float(np.linalg.norm(width/2))
        centers=np.asarray([low+(np.asarray(index)+.5)*width for index in np.ndindex(tuple(counts))])
        keep=trimesh.proximity.signed_distance(mesh,centers)>=-radius
        centers=centers[keep];signed=trimesh.proximity.signed_distance(mesh,centers)
        radii=np.maximum(radius,signed+.0002)
        np.random.seed(0);points=np.r_[mesh.vertices,mesh.sample(10000),centers[signed>0]]
        covers=cdist(centers,points)<=radii[:,None]-.00005;remaining=np.ones(len(points),dtype=bool);parts=[]
        while remaining.any():
            score=covers[:,remaining].sum(1);index=int(score.argmax())
            if score[index]==0:raise ValueError("Uncovered collision geometry")
            parts.append({"center":centers[index].tolist(),"radius":float(radii[index])});remaining[covers[index]]=False
        np.random.seed(1);surface=mesh.sample(30000)
        while True:
            gap=np.min(cdist(surface,np.asarray([p["center"] for p in parts]))-np.asarray([p["radius"] for p in parts])[None,:],axis=1)
            index=int(gap.argmax())
            if gap[index]<=.0002:break
            parts.append({"center":surface[index].tolist(),"radius":.005})
        spheres["collision_spheres"][name]=parts
    sphere_path.write_text(yaml.safe_dump(spheres,sort_keys=False))
    if sum(map(len,spheres["collision_spheres"].values()))>=1024:
        raise ValueError("cuRobo self-collision kernel requires fewer than 1024 spheres")
    config=yaml.safe_load((new/"config.yml").read_text())
    actions={side:load_actions(episode,side) for side in ("left","right")}
    config["homestate"]=[actions[s]["joint_positions"][0].tolist() for s in ("left","right")]
    (new/"config.yml").write_text(yaml.safe_dump(config,sort_keys=False))
    path=ROOT/"env_cfg/task_config/_embodiment_config.yml"
    registered=yaml.safe_load(path.read_text());registered["w1-g2-black"]={"file_path":"./assets/embodiments/w1-g2-black"}
    path.write_text(yaml.safe_dump(registered,sort_keys=False))
    assets=yaml.safe_load((ROOT/"env_cfg/w1_finals_assets_meshy_v1.yml").read_text())
    ratio=.26/.28
    assets["pan"]["scale"]=ratio
    for key in ("inner_radius","floor_height","rim_height"):assets["pan"][key]*=ratio
    assets["pan"]["mouth_diameter_m"]=.26
    assets["green_pepper"]["initial_xy"]=[-.38,.045]
    assets["green_pepper"]["pan_offset_xy"]=[-.025,-.012]
    assets["green_pepper"]["turn_heading"]=2.2
    assets["small_pepper"]=copy.deepcopy(assets["green_pepper"])
    assets["small_pepper"].update(scale=.065/.11,mass=.012,initial_xy=[.439,-.0004],grasp_offset=[-.012,0,.004])
    assets["potato"].update(initial_xy=[.413,-.126],scale=.036)
    assets["eggplant"].update(initial_xy=[-.4225,-.202])
    start=json.loads((Path(episode)/"meta.json").read_text())["created_at"]
    # Real EE quaternions are xyzw and rotate 90 degrees around forward from
    # RoboTwin's virtual TCP. This conversion was checked against URDF FK.
    for kind,side,time in (("green_pepper","left",24.3),("small_pepper","right",7.3),
                           ("potato","right",38.5),("eggplant","left",60.8)):
        a=actions[side];i=np.argmin(abs(a["timestamp"]-start-time))
        rot=Rotation.from_euler("z",np.pi/2)*Rotation.from_quat(a["ee_positions"][i,3:])*Rotation.from_euler("x",-np.pi/2)
        assets[kind]["real_grasp_rotation_rpy"]=rot.as_euler("xyz").tolist()
        assets[kind]["preferred_grasp_rotation_rpy"]=(rot.as_euler("xyz").tolist() if kind=="potato" else [-float(np.pi/2),float(np.pi/2),0.0])
    folder=ROOT/"assets/objects/w1_real_reference_v1";folder.mkdir(parents=True,exist_ok=True)
    prepare_pan_collision(assets,folder)
    tool=json.loads((ROOT/assets["spatula"]["metadata"]).read_text())
    tool.update(rest_yaw=float(np.pi/2),rest_height=.007,use_stand=False,grasp_offset=[0,.035,.006],
                grasp_point_tool_local=[.035,0,.006],
                lift_vector=[0,0,.13],approach_offset=[0,.035,.13],
                enter_pitch=.65,enter_roll=0.,second_enter_roll=0.,hover_pitch=0.,
                scoop_pitch=.4,scoop_lift=.02,scoop_advance=.01,push_distance=.05,
                second_push_distance=.05,tip_clearance=.012,second_tip_clearance=.012,
                second_scoop_pitch=.4,second_scoop_roll=0.,second_scoop_lift=.035)
    tool.pop("contact_x_max_from_pan",None);tool.pop("contact_x_min_from_pan",None)
    a=actions["right"];i=np.argmin(abs(a["timestamp"]-start-75.7))
    rot=Rotation.from_euler("z",np.pi/2)*Rotation.from_quat(a["ee_positions"][i,3:])*Rotation.from_euler("x",-np.pi/2)
    tool["grasp_rotation_rpy"]=rot.as_euler("xyz").tolist()
    (folder/"spatula_metadata.json").write_text(json.dumps(tool,indent=2)+"\n")
    assets["spatula"]["metadata"]=str((folder/"spatula_metadata.json").relative_to(ROOT))
    manifest=ROOT/"env_cfg/w1_real_reference_assets_v1.yml"
    manifest.write_text(yaml.safe_dump(assets,sort_keys=False))
    cfg=yaml.safe_load((ROOT/"env_cfg/task_config/w1_finals_meshy_v2.yml").read_text())
    cfg.update(embodiment=["w1-g2-black","w1-g2-black",.6],asset_manifest=str(manifest.relative_to(ROOT)),
               pan_position_xy=[-.04,-.12],tool_rest_xy=[.2725,-.166],tool_arm="right",
               real_episode=str(Path(episode).resolve().relative_to(ROOT)),max_seed_attempts=20,
               tool_cartesian_steps=6,recording_profile="w1_native_v0.1",render_mode="raster")
    cfg.update(small_object_contact_offset_m=.002,small_object_solver_iterations=[32,8],
               final_stability_min_s=1.0,final_settle_timeout_s=15.0)
    cfg["data_frequency"]=30;cfg["data_type"]["depth"]=True
    cfg["camera"].update(head_camera_type="W1_REAL_640",wrist_camera_type="W1_REAL_640")
    (ROOT/"env_cfg/task_config/w1_finals_real_v1.yml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    p=ROOT/"env_cfg/task_config/_camera_config.yml";cams=yaml.safe_load(p.read_text())
    cams["W1_REAL_640"]={"w":640,"h":480,"fovy":43.999}
    p.write_text(yaml.safe_dump(cams,sort_keys=False))
    (folder/"adaptation.json").write_text(json.dumps({"source_episode":str(episode),"pan_diameter_m":.26,
        "base_separation_m":.6,"arm_color":"black","potato_length_estimate_m":.0684,"small_pepper_length_estimate_m":.065,
        "inferred_sizes_not_measurements":["potato","small_pepper"],"original_robot_and_glb_files_modified":False,
        "wrist_calibration":"not supplied; provisional CAD mounts","pan_dynamics":"fixed during initial collection profile"},indent=2)+"\n")
    print("Prepared w1_finals_real_v1; original assets and datasets retained")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("episode",type=Path,nargs="?",default=ROOT/"data/real_w1/episode_0001")
    prepare(p.parse_args().episode)
