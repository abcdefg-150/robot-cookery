#!/usr/bin/env python3
"""Install the W1 overlay into an existing RoboTwin checkout, after preflight."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

PACKAGE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_apply(target, patch, *options):
    environment=dict(os.environ,GIT_CEILING_DIRECTORIES=str(target.parent))
    result=subprocess.run(['git', 'apply', '--verbose', *options, str(patch)], cwd=target,
                          env=environment,text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if 'Skipped patch' in result.stderr:raise RuntimeError(f'Patch was skipped rather than checked: {patch}')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();target=args.target.expanduser().resolve()
    manifest=json.loads((PACKAGE/'manifest.json').read_text())
    for relative,expected in manifest['package_sha256'].items():
        path=PACKAGE/relative
        if not path.is_file() or digest(path)!=expected:raise RuntimeError(f'Package integrity failure: {relative}')
    for relative in ('envs/_base_task.py','envs/robot/planner.py','XPolicyLab/utils/process_data.py'):
        if not (target/relative).is_file():raise RuntimeError(f'RoboTwin/XPolicyLab dependency missing: {relative}')
    state_path=target/'.w1_cookery_install.json'
    previous=json.loads(state_path.read_text()).get('installed_sha256',{}) if state_path.exists() else {}
    copies=[];unchanged=[]
    for source in sorted((PACKAGE/'overlay').rglob('*')):
        if not source.is_file():continue
        relative=source.relative_to(PACKAGE/'overlay');destination=target/relative
        if not destination.resolve().is_relative_to(target):raise RuntimeError(f'Destination escapes target through a symlink: {relative}')
        if destination.exists():
            if str(relative) in manifest.get('base_resource_defaults',[]):unchanged.append(str(relative));continue
            if digest(destination)==digest(source):unchanged.append(str(relative));continue
            if previous.get(str(relative))!=digest(destination):
                raise RuntimeError(f'Existing file differs; preflight stopped before writing: {relative}')
        copies.append((source,destination))
    patches=[]
    for entry in manifest['patches']:
        patch=PACKAGE/entry['file'];workdir=target/entry.get('cwd','.')
        reverse=git_apply(workdir,patch,'--check','--reverse')
        if reverse.returncode==0:continue
        forward=git_apply(workdir,patch,'--check')
        if forward.returncode:raise RuntimeError(f'Core patch conflict: {entry["file"]}\n{forward.stderr}')
        patches.append((patch,workdir,entry['targets']))
    registries=[]
    for relative,entries in manifest['registry_entries'].items():
        path=target/relative
        original=path.read_text()
        data=json.loads(original) if path.suffix=='.json' else yaml.safe_load(original)
        additions={}
        for key,value in entries.items():
            if key in data and data[key]!=value:raise RuntimeError(f'Registry entry conflict: {relative}:{key}')
            if key not in data:additions[key]=value
        if not additions:continue
        if path.suffix=='.json':
            data.update(additions);updated=json.dumps(data,indent=2)+'\n'
        else:updated=original.rstrip()+'\n'+yaml.safe_dump(additions,sort_keys=False)
        registries.append((path,updated))
    runtime_configs=[]
    for template in (PACKAGE/'overlay/assets/embodiments').glob('*/curobo_tmp.yml'):
        relative=template.relative_to(PACKAGE/'overlay')
        path=(target/relative).with_name('curobo.yml')
        updated=template.read_text().replace('${ASSETS_PATH}',str(target))
        if path.exists() and yaml.safe_load(path.read_text())==yaml.safe_load(updated):continue
        runtime_configs.append((path,updated))
    summary={'target':str(target),'dry_run':args.dry_run,'files_to_copy':len(copies),
             'core_patches':len(patches),'registries_to_merge':len(registries),
             'runtime_planner_configs':len(runtime_configs),'identical_files':len(unchanged)}
    print(json.dumps(summary,indent=2))
    if args.dry_run:return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=target/'.w1_install_backups'/stamp
    touched={p for _,p in copies}|{p for p,_ in registries}|{p for p,_ in runtime_configs}
    for _,workdir,paths in patches:touched.update(workdir/p for p in paths)
    for path in touched:
        if path.exists():
            save=backup/path.relative_to(target);save.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,save)
    for patch,workdir,_ in patches:
        result=git_apply(workdir,patch)
        if result.returncode:raise RuntimeError(f'Patch failed after successful preflight: {result.stderr}; backup={backup}')
    for source,destination in copies:
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    for path,value in registries:path.write_text(value)
    for path,value in runtime_configs:path.write_text(value)
    installed={str(path.relative_to(target)):digest(path) for path in touched}
    installed.update({name:digest(target/name) for name in unchanged})
    state={'package':manifest['release'],'backup':str(backup),'installed_sha256':installed,'summary':summary}
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    print(f'Installed {manifest["release"]}. Existing modified files were backed up to {backup}')


if __name__=='__main__':main()
