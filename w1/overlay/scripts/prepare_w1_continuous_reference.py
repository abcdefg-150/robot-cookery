"""Keep the real scene layout and prepare the continuous-motion W1 profile."""
from pathlib import Path
import json,shutil,yaml,xml.etree.ElementTree as E
ROOT=Path(__file__).resolve().parents[1]
def prepare():
 old=ROOT/"assets/embodiments/w1-g2-black";new=ROOT/"assets/embodiments/w1-g2-black-real-v2";new.mkdir(parents=True,exist_ok=True)
 for source in old.iterdir():
  if source.is_file():shutil.copy2(source,new/source.name)
 for p in new.glob("*.yml"):
  p.write_text(p.read_text().replace("/w1-g2-black/","/w1-g2-black-real-v2/"))
 p=new/"w1_g2.urdf";tree=E.parse(p)
 for link in tree.getroot().findall("link"):
  for visual in link.findall("visual"):
   material=visual.find("material")
   if material is not None:
    material.set("name","w1_render_"+link.get("name"));material.find("color").set("rgba","0.40 0.42 0.43 1" if link.get("name")=="g2_base_link" else "0.035 0.04 0.045 1")
 E.indent(tree,space="  ");tree.write(p,encoding="utf-8",xml_declaration=True)
 p=new/"config.yml";cfg=yaml.safe_load(p.read_text());cfg["gripper_stiffness"]=3000;cfg["gripper_damping"]=200;p.write_text(yaml.safe_dump(cfg,sort_keys=False))
 for p in (new/"curobo.yml",new/"curobo_tmp.yml"):
  cfg=yaml.safe_load(p.read_text());cfg["planner"].update(continuous_urdf_trajectory=True,trajectory_velocity_limits=[.95,.93,.98,.95,1.45,.66],trajectory_acceleration_limits=[3.]*6);p.write_text(yaml.safe_dump(cfg,sort_keys=False))
 p=ROOT/"env_cfg/task_config/_embodiment_config.yml";cfg=yaml.safe_load(p.read_text());cfg["w1-g2-black-real-v2"]={"file_path":"./assets/embodiments/w1-g2-black-real-v2"};p.write_text(yaml.safe_dump(cfg,sort_keys=False))
 a=yaml.safe_load((ROOT/"env_cfg/w1_real_reference_assets_v1.yml").read_text());tool=json.loads((ROOT/a["spatula"]["metadata"]).read_text());tool["lift_segments"]=1
 p=ROOT/"assets/objects/w1_real_reference_v1/spatula_metadata_v2.json";p.write_text(json.dumps(tool,indent=2)+"\n");a["spatula"]["metadata"]=str(p.relative_to(ROOT));p=ROOT/"env_cfg/w1_real_reference_assets_v2.yml";p.write_text(yaml.safe_dump(a,sort_keys=False))
 c=yaml.safe_load((ROOT/"env_cfg/task_config/w1_finals_real_v1.yml").read_text());c.update(embodiment=["w1-g2-black-real-v2","w1-g2-black-real-v2",.6],asset_manifest=str(p.relative_to(ROOT)),tool_cartesian_steps=3)
 # Positions are deliberately retained from the real-camera layout fit.
 (ROOT/"env_cfg/task_config/w1_finals_real_v2.yml").write_text(yaml.safe_dump(c,sort_keys=False))
 print("Prepared continuous profile; real pan/spatula placement retained")
if __name__=="__main__":prepare()
