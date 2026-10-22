#!/usr/bin/env python3
"""Paper-aligned interfaces with explicitly named G1/project adaptations (MBRL)."""
import os,sys,json,time,math,argparse,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor'))
from mbrl.runtime import configure_runtime
configure_runtime()
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import torch
from mbrl.core import WorldModel,Actor,Critic,frozen,world_loss,imagine,rollout_support
from mbrl.adapter import Adapter,Replay


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['tracking'],default='tracking')
    p.add_argument('--profile',choices=['paper','small'],default='small')
    p.add_argument('--motion',default='');p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--iterations',type=int,default=320000)
    p.add_argument('--seconds',type=float,default=3600)
    for flag in ['num-envs','model-batch','policy-batch']:
        p.add_argument('--'+flag,type=int)
    p.add_argument('--wm-width',type=int,default=2048)
    p.add_argument('--wm-batches',type=int,default=10)
    p.add_argument('--horizon',type=int,default=32)
    p.add_argument('--history',type=int,default=10)
    p.add_argument('--goal-frames',type=int,default=10)
    p.add_argument('--window',type=int,default=8)
    p.add_argument('--collect',type=int,default=4)
    p.add_argument('--replay',type=int,default=1000)
    p.add_argument('--action-noise',type=float,default=.1)
    p.add_argument('--noise-scale',type=float,default=1.)
    p.add_argument('--max-state-z',type=float,default=float('inf'))
    p.add_argument('--simple-standing',action='store_true')
    p.add_argument('--no-return-sampling',action='store_true')
    p.add_argument('--resume',default='')
    args=p.parse_args()
    defaults=(4096,2048,4096) if args.profile=='paper' else (256,512,512)
    for key,value in zip(['num_envs','model_batch','policy_batch'],defaults):
        if getattr(args,key) is None:setattr(args,key,value)
    if min(args.num_envs-1,args.model_batch,args.policy_batch,args.window,args.horizon,args.collect)>0 and args.replay>args.window:
        return args
    p.error('Invalid batch/environment/window sizes')


def main():
    args=parse_args();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    ad=Adapter(args.kind,args.motion,args.num_envs,args.seed,args.history,
               simple=args.simple_standing,return_sampling=not args.no_return_sampling)
    ad.noise*=args.noise_scale
    ad.h=ad.noisy(ad.obs['actor'])[:,None].repeat(1,args.history,1)
    (out/'schema.json').write_text(json.dumps(ad.schema,indent=2))
    s,c=ad.state(),ad.context();g,_=ad.goal()
    dims=dict(ns=s.shape[-1],na=ad.na,ng=g.shape[-1],nc=c.shape[-1],no=ad.no,nf=ad.nf)
    wm=WorldModel(**dims,widths=(args.wm_width,)*3,analytic_proprio=True).cuda()
    wm.default_pose.copy_(ad.default_pose)
    actor=Actor(ad.no,ad.na,args.history,dims['ng'],args.goal_frames,args.profile).cuda()
    critic=Critic(wm).cuda()
    with torch.no_grad():wm.reward_done[-1].bias[1]=-4
    modules=[wm,actor,critic]
    opts=[torch.optim.Adam(wm.parameters(),lr=2e-4),
          torch.optim.Adam(actor.parameters(),lr=2e-4,betas=(.9,.95)),
          torch.optim.Adam(critic.parameters(),lr=2e-4)]
    counters=dict(model_updates=0,policy_updates=0,critic_updates=0,
                  rejected_trajectories=0,proposed_trajectories=0,empty_policy_batches=0)
    first=0
    if args.resume:
        ck=torch.load(args.resume,weights_only=False)
        if ck.get('format_version')!=3:raise ValueError('Old checkpoint interface incompatible; use MBRL checkpoint')
        for module,sd in zip(modules,ck['modules']):module.load_state_dict(sd)
        for opt,sd in zip(opts,ck['optimizers']):opt.load_state_dict(sd)
        first=ck['iteration']+1;counters=ck['counters']
        (out/'resume_check.json').write_text(json.dumps(dict(source=args.resume,
            module_weights_exact=all(torch.equal(m.state_dict()[k],v) for m,sd in zip(modules,ck['modules']) for k,v in sd.items()),
            optimizer_state_entries=[len(o.state) for o in opts],first_iteration=first,
            replay_and_simulator_restarted=True)))
        if ad.return_sampler is not None and ck.get('sampler') is not None:
            for name in ['values','past']:getattr(ad.return_sampler,name).copy_(ck['sampler'][name])
            ad.return_sampler.updates=ck['sampler']['updates']
    replay=Replay(args.replay);clock=time.monotonic()
    for _ in range(max(64,args.window+1)):replay.add(ad.step(actor,args.action_noise))
    if not args.resume:
        b=replay.sample(1,min(8192,args.num_envs*32))
        for norm,key in [(wm.sn,'s'),(wm.gn,'g'),(wm.cn,'c'),(wm.on,'o'),(actor.normalizer,'o')]:norm.fit(b[key][0])
        actor.goal_normalizer.fit(ad.goal_table)
        wm.rn.fit(b['r'][0,:,None])
    def save(name,it):
        ck=dict(format_version=3,args=vars(args),dims=dims,iteration=it,
                modules=[m.state_dict() for m in modules],optimizers=[o.state_dict() for o in opts],
                counters=counters,sampler=ad.return_sampler.state_dict() if ad.return_sampler else None,
                elapsed=time.monotonic()-clock,replay_saved=False,
                resume_note='Resume weights/optimizers/sampler; simulator and replay restart, not bitwise continuation')
        temp=out/(name+'.tmp');torch.save(ck,temp);temp.replace(out/name)
    save('initial.pt',first-1)
    logs=(out/'metrics.jsonl').open('w',buffering=1)
    try:
        for it in range(first,first+args.iterations):
            lr_factor=.03+.97*.5*(1+math.cos(math.pi*min(it,250000)/250000))
            for i,opt in enumerate(opts):
                for group in opt.param_groups:group['lr']=2e-4*lr_factor
            real=[]
            for _ in range(args.collect):
                row=ad.step(actor,args.action_noise);replay.add(row);real.append(row['r'].mean().item())
            for _ in range(args.wm_batches):
                loss,err=world_loss(wm,replay.sample(args.window,args.model_batch))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite model loss')
                opts[0].zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.parameters(),1.,error_if_nonfinite=True);opts[0].step()
                counters['model_updates']+=1
            save('final.pt',it)
        (out/'status.json').write_text(json.dumps(dict(status='finished',**counters)))
    except Exception as exc:
        (out/'status.json').write_text(json.dumps(dict(status='failed',error=str(exc),**counters)))
        raise
    finally:logs.close();ad.env.close()


if __name__=='__main__':main()
