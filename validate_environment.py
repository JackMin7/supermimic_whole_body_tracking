"""Integration contract checks against MuJoCo physical state, no policy training."""
import os,sys,json,argparse
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'vendor'))
from mbrl.runtime import configure_runtime
configure_runtime()
os.environ.setdefault('MUJOCO_GL','egl')
import torch
from mbrl.adapter import Adapter,Replay
from mbrl.core import Actor,WorldModel,world_loss
from mjlab.utils.lab_api.math import quat_apply_inverse

p=argparse.ArgumentParser();p.add_argument('--motion',required=True);p.add_argument('--output',required=True)
a=p.parse_args();torch.set_num_threads(4)
ad=Adapter('tracking',a.motion,8,42,10,simple=True)
robot=ad.env.scene['robot'];cmd=ad.env.command_manager.get_term('motion')
actor=Actor(93,29,10,ad.goal()[0].shape[-1],10).cuda()
replay=Replay(32);checks={};max_ang=max_gravity=max_joint=max_fk=0.
# Independent entity observations vs analytical qpos/qvel reconstruction.
for i in range(20):
    s=ad.state();o=ad.proprio();c=ad.context()
    idx=robot.body_names.index('pelvis')
    quat=robot.data.body_link_quat_w[:,idx]
    ang=quat_apply_inverse(quat,robot.data.body_link_ang_vel_w[:,idx])
    gravity=quat_apply_inverse(quat,robot.data.gravity_vec_w)
    max_ang=max(max_ang,float((o[:,58:61]-ang).abs().max()))
    max_gravity=max(max_gravity,float((o[:,61:64]-gravity).abs().max()))
    max_joint=max(max_joint,float((s[:,7:36]-robot.data.joint_pos).abs().max()))
    if i==0:
        max_fk=float((cmd.body_pos_w-cmd.robot_body_pos_w).abs().max())
    replay.add(ad.step(actor,.1))
checks.update(max_angular_observation_error=max_ang,max_gravity_observation_error=max_gravity,
              max_joint_order_error=max_joint,initial_reference_body_position_error=max_fk,
              anchor=cmd.cfg.anchor_body_name,observation_dim=ad.no,reward_scale=ad.reward_scale)
assert max_ang<1e-4 and max_gravity<1e-4 and max_joint<1e-5,checks
assert cmd.cfg.anchor_body_name=='pelvis' and ad.no==93 and ad.reward_scale==1
s,c=ad.state(),ad.context();g,_=ad.goal()
wm=WorldModel(s.shape[-1],29,g.shape[-1],c.shape[-1],93,widths=(64,64),analytic_proprio=True).cuda()
wm.default_pose.copy_(ad.default_pose)
b=replay.sample(8,8);loss,_=world_loss(wm,b);loss.backward()
assert torch.isfinite(loss),loss
checks['world_loss_backward_finite']=True
Path(a.output).write_text(json.dumps(checks,indent=2));print(checks);ad.env.close()
