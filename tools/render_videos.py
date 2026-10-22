import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor'))
os.environ['MUJOCO_GL']='egl'
import numpy as np
import torch,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from mbrl.runtime import configure_runtime
configure_runtime()
from mbrl.adapter import Adapter
import argparse
p=argparse.ArgumentParser()
p.add_argument('--motion',required=True);p.add_argument('--results',default=str(ROOT/'results/standing_1h'));p.add_argument('--output',required=True)
args=p.parse_args();results=Path(args.results)
motion=args.motion
torch.set_num_threads(4)
ad=Adapter('tracking',motion,1,1234,10,evaluation=True,simple=True,return_sampling=False)
model=ad.env.sim.mj_model
model.vis.global_.offwidth=640;model.vis.global_.offheight=480
renderer=mujoco.Renderer(model,height=480,width=640)
data=mujoco.MjData(model)
cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:]=[0,0,.65];cam.distance=2.7;cam.azimuth=130;cam.elevation=-15
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
except OSError:font=ImageFont.load_default()
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
meta={'source':'Recorded physical-simulator trajectories, not generated animation','replica':0,'training_minutes':60,'fps':25,'failure_display':'Freeze last recorded pre-termination frame; no reset replay'}
for suffix,duration,label in [('final',20,'20s'),('long60',60,'60s')]:
    arrays={n:np.load(results/f'{n}_{suffix}_trajectory.npz') for n in ['ppo','mbrl']}
    traj={n:v['trajectory'] for n,v in arrays.items()}
    failures={n:(int(ix[0]) if len(ix:=np.flatnonzero(t[:,-1])) else None) for n,t in traj.items()}
    meta[label]={n:None if failures[n] is None else (failures[n]+1)*.02 for n in traj}
    writers={n:imageio.get_writer(str(out/f'{n}_{label}.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=1) for n in ['ppo','mbrl','comparison']}
    for frame in range(duration*25):
        seconds=frame/25;index=round(seconds/.02)
        panels=[]
        for name,title,color in [('ppo','PPO (Original BeyondMimic)',(70,130,220)),('mbrl','MBRL (SuperMimic)',(230,150,50))]:
            failed=failures[name] is not None and index>=failures[name]
            i=min(index,failures[name]) if failed else index
            row=traj[name][i]
            data.qpos[:]=row[:36];data.qvel[:]=row[36:71]
            mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=cam)
            panel=Image.new('RGB',(640,576),(22,26,33))
            panel.paste(Image.fromarray(renderer.render()),(0,80))
            draw=ImageDraw.Draw(panel)
            draw.text((16,8),f'{title} | 1 h training',font=font,fill=color)
            draw.text((16,37),f'Time {seconds:05.2f} s | evaluation environment 0',font=small,fill='white')
            state=f'TERMINATED at {(failures[name]+1)*.02:.2f} s - FROZEN' if failed else 'No termination yet'
            draw.text((16,59),state,font=small,fill=(255,100,100) if failed else (120,235,150))
            draw.text((16,558),'Recorded simulation replay | no sensor noise or pushes',font=small,fill=(200,205,215))
            panels.append(panel);writers[name].append_data(np.asarray(panel))
        combined=Image.new('RGB',(1280,576));combined.paste(panels[0],(0,0));combined.paste(panels[1],(640,0))
        writers['comparison'].append_data(np.asarray(combined))
        if suffix=='final' and frame==100:combined.save(out/'comparison_preview.png')
    for w in writers.values():w.close()
(out/'video_metadata.json').write_text(json.dumps(meta,indent=2))
renderer.close();ad.env.close()
