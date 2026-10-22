"""Evaluate only MBRL checkpoints; old observation contracts are rejected explicitly."""
import os,sys,json,argparse
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'vendor'))
from mbrl.runtime import configure_runtime
configure_runtime()
os.environ.setdefault('MUJOCO_GL','egl')
import torch
import numpy as np
from mbrl.adapter import Adapter
from mbrl.core import Actor
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
p.add_argument('--output',required=True);p.add_argument('--seconds',type=float,default=20)
p.add_argument('--num-envs',type=int,default=32);p.add_argument('--motion',default='')
a=p.parse_args();torch.set_num_threads(4)
ck=torch.load(a.checkpoint,weights_only=False)
if ck.get('format_version')!=3:raise ValueError('Expected MBRL checkpoint')
c=ck['args'].copy()
if a.motion:c['motion']=a.motion
ad=Adapter(c['kind'],c['motion'],a.num_envs,1234,c['history'],evaluation=True,simple=True,return_sampling=False)
# A long standing test must not teleport/reset at the original 30-second clip end.
if c['kind']=='tracking':
    cmd=ad.env.command_manager.get_term('motion');m=cmd.motion
    required=round(a.seconds/ad.env.step_dt)+c['goal_frames']+2
    if required>m.time_step_total:
        fields=['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']
        for name in fields:
            value=getattr(m,name)
            if (value-value[:1]).abs().max()>1e-5:
                raise ValueError('Long evaluation extension is only defined for static references')
            setattr(m,name,value[:1].expand(required,*value.shape[1:]).clone())
        ad.goal_table=ad.goal_table[:1].expand(required,-1).clone();m.time_step_total=required
ad.noise.zero_();ad.h=ad.obs['actor'][:,None].repeat(1,c['history'],1)
actor=Actor(ad.no,ad.na,c['history'],ck['dims']['ng'],c['goal_frames'],c['profile']).cuda()
actor.load_state_dict(ck['modules'][1]);actor.eval()
alive=torch.ones(a.num_envs,device='cuda:0',dtype=torch.bool)
errors=[];trajectory=[]
duration=torch.zeros(a.num_envs,device='cuda:0');returns=duration.clone()
with torch.no_grad():
    for _ in range(round(a.seconds/ad.env.step_dt)):
        row=ad.step(actor,0)
        duration+=alive*ad.env.step_dt;returns+=alive*row['r']
        error=(row['nxt'][:,7:36]-row['g'][:,:29]).square().mean(-1).sqrt()
        if alive.any():errors.append(float(error[alive].mean()))
        trajectory.append(torch.cat((row['s'][0],row['a'][0],row['term'][0:1])).cpu().numpy())
        alive &= ~row['term'].bool()
result=dict(mean_time_before_failure_s=float(duration.mean()),survivors=int(alive.sum()),
            environments=a.num_envs,task_return=float(returns.mean()),duration_s=a.seconds,
            joint_rmse_before_failure_rad=sum(errors)/len(errors) if errors else None,
            note='Nominal simulation diagnostic, not independent seeds or real-robot validation')
np.savez_compressed(str(Path(a.output).with_suffix('.npz')),trajectory=np.array(trajectory),dt=ad.env.step_dt)
Path(a.output).write_text(json.dumps(result,indent=2));print(result);ad.env.close()
