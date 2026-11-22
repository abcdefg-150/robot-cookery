"""Prepare a separate image-matched scene, retaining previous size profiles."""
from pathlib import Path
import json,copy,yaml,numpy as np,trimesh,shutil
from scipy.spatial.transform import Rotation as R
ROOT=Path(__file__).resolve().parents[1]
def prepare():
 fit=json.loads((ROOT/"env_cfg/w1_real_image_alignment_v1.json").read_text());a=yaml.safe_load((ROOT/"env_cfg/w1_real_reference_assets_v2.yml").read_text());old_pan_scale=a["pan"]["scale"]
 x,y,scale=fit["pan"]["fit_xy_scale"];ratio=scale/old_pan_scale;pan=a["pan"];pan["scale"]=scale
 for key in ("inner_radius","floor_height","rim_height"):pan[key]*=ratio
 profile=pan["collision_approximation"]
 for key in ("radii_m","heights_m"):profile[key]=(np.array(profile[key])*ratio).tolist()
 profile["outer_radius_m"]=.132*ratio
 pan.update(user_reported_diameter_m=.26,diameter_measurement_basis="unconfirmed; fitted to RGB image",simulated_outer_diameter_m=fit["pan"]["outer_width_m"],mouth_diameter_m=2*profile["radii_m"][-1])
 for kind,f in fit["foods"].items():
  xk,yk,factor,yaw=f["xy_scale_yaw"];cfg=a[kind];cfg["previous_nominal_scale"]=cfg["scale"];cfg["scale"]*=factor;cfg["initial_xy"]=[xk,yk]
  rot=R.from_euler("z",yaw);cfg["rotation_rpy"]=(rot*R.from_euler("xyz",cfg["rotation_rpy"])).as_euler("xyz").tolist()
  cfg["grasp_offset"]=(rot.apply(np.array(cfg.get("grasp_offset",[0,0,0]))*factor)).tolist()
  cfg["rescale_inertia_to_mass"]=True;cfg["image_fitted_dimensions_not_measured"]=f["estimated_visual_dimensions_m"]
 folder=ROOT/"assets/objects/w1_real_image_v1";folder.mkdir(parents=True,exist_ok=True)
 tx,ty,yaw,width=fit["spatula"]["xy_yaw_blade_width_factor"];tool=json.loads((ROOT/a["spatula"]["metadata"]).read_text())
 def warp(v):
  v=np.asarray(v).copy();blend=np.clip((v[:,0]-.10)/.03,0,1);v[:,1]*=1+(width-1)*blend;return v
 mesh=trimesh.load(ROOT/a["spatula"]["visual"],force="mesh");mesh.vertices=warp(mesh.vertices);p=folder/"spatula_visual.glb";mesh.export(p);a["spatula"]["visual"]=str(p.relative_to(ROOT))
 for i,part in enumerate(tool["parts"]):
  if "collision_mesh" in part:
   mesh=trimesh.load(ROOT/part["collision_mesh"],force="mesh");mesh.vertices=warp(mesh.vertices);p=folder/f"spatula_collision_{i:03d}.obj";mesh.export(p);part["collision_mesh"]=str(p.relative_to(ROOT))
 tool.update(rest_yaw=yaw,lift_segments=1,rescale_inertia_to_mass=True,grasp_point_tool_local=[.020,0,.006],grasp_offset=(R.from_euler("z",yaw).apply([.020,0,.006])).tolist(),blade_width_image_fit_factor=width,enter_pitch=.8,hover_pitch=.8,push_distance=.035,second_push_distance=.035,scoop_lift=.04,second_scoop_lift=.04)
 a["green_pepper"]["turn_heading"]=.8
 p=folder/"spatula_metadata.json";p.write_text(json.dumps(tool,indent=2)+"\n");a["spatula"]["metadata"]=str(p.relative_to(ROOT))
 p=ROOT/"env_cfg/w1_real_reference_assets_v3.yml";p.write_text(yaml.safe_dump(a,sort_keys=False))
 cfg=yaml.safe_load((ROOT/"env_cfg/task_config/w1_finals_real_v2.yml").read_text());cfg.update(asset_manifest=str(p.relative_to(ROOT)),pan_position_xy=[x,y],tool_rest_xy=[tx,ty],wrist_image_fit="env_cfg/w1_wrist_image_fit_v1.json",tool_cartesian_steps=1)
 (ROOT/"env_cfg/task_config/w1_finals_real_v3.yml").write_text(yaml.safe_dump(cfg,sort_keys=False))
 print("Prepared image-matched v3; older metric and asset profiles retained")
if __name__=="__main__":prepare()
