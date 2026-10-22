"""Native installed RSL-RL PPO with matched MBRL actor/critic and standing environment."""
import os,sys,sysconfig
from pathlib import Path
sys.path.insert(0,sysconfig.get_paths()['purelib'])
import rsl_rl
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor'))
from mbrl.runtime import configure_runtime
configure_runtime()
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,time
from dataclasses import asdict
from itertools import chain
import torch
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mbrl.adapter import Adapter
from mbrl.core import Actor


class PackedActor(torch.nn.Module):
    def __init__(self,actor,history,no,ng,k):
        super().__init__();self.actor=actor;self.history=history;self.no=no;self.ng=ng;self.k=k
    def forward(self,x):
        return self.actor(x[:,:self.history*self.no].reshape(-1,self.history,self.no),
                          x[:,self.history*self.no:].reshape(-1,self.k,self.ng))


class MatchedWrapper(RslRlVecEnvWrapper):
    def __init__(self,ad,shared,k):
        self.ad,self.shared,self.k=ad,shared,k
        super().__init__(ad.env,clip_actions=None)
    def _align(self,obs,advance=False,done=None):
        if advance:
            frame=self.ad.proprio()
            self.ad.h=torch.cat((self.ad.h[:,1:],frame[:,None]),1)
            if done is not None:self.ad.h[done.bool()]=frame[done.bool(),None].repeat(1,self.ad.history,1)
        g,meta=self.ad.goal()
        future=self.ad.goals(meta,self.k-1).transpose(0,1)
        obs['actor']=torch.cat((self.ad.h.flatten(1),future.flatten(1)),-1)
        tensors=[self.ad.state(),g,self.ad.context()]
        obs['critic']=torch.cat([(x-self.shared[p+'.mean'])/self.shared[p+'.scale']
                                 for p,x in zip(['sn','gn','cn'],tensors)],-1)
        return obs
    def get_observations(self):return self._align(super().get_observations())
    def step(self,action):
        obs,r,d,info=super().step(action)
        return self._align(obs,True,d),r,d,info


class Deadline(Exception):pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--motion',required=True)
    p.add_argument('--seconds',type=float,default=600);p.add_argument('--num-envs',type=int,default=256)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--output',required=True)
    p.add_argument('--common-init',required=True);p.add_argument('--resume',default='');args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=True
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+180
    while not Path(args.common_init).exists():
        if time.monotonic()>deadline:raise TimeoutError('Missing shared initialization')
        time.sleep(.2)
    ck=torch.load(args.common_init,weights_only=False);c=ck['args'];dims=ck['dims']
    ad=Adapter('tracking',args.motion,args.num_envs,args.seed,c['history'],simple=True,return_sampling=False)
    ad.noise.zero_();ad.h=ad.obs['actor'][:,None].repeat(1,c['history'],1)
    cfg=asdict(unitree_g1_tracking_ppo_runner_cfg())
    cfg.update(seed=args.seed,logger='tensorboard',upload_model=False,save_interval=100)
    cfg['actor']['hidden_dims']=(512,256,128);cfg['actor']['obs_normalization']=False
    cfg['actor']['distribution_cfg']['init_std']=.1
    cfg['critic']['hidden_dims']=(256,128);cfg['critic']['obs_normalization']=False
    env=MatchedWrapper(ad,ck['modules'][0],c['goal_frames'])
    runner=MjlabOnPolicyRunner(env,cfg,str(out),'cuda:0')
    actor=Actor(ad.no,ad.na,c['history'],dims['ng'],c['goal_frames'],c['profile']).cuda()
    actor.load_state_dict(ck['modules'][1])
    policy=runner.alg.get_policy()
    policy.mlp=PackedActor(actor,c['history'],ad.no,dims['ng'],c['goal_frames'])
    runner.alg.critic.mlp.load_state_dict({k[4:]:v for k,v in ck['modules'][2].items() if k.startswith('net.')})
    # Rebuild the native optimizer after replacing its actor network so encoder and
    # decoder parameters are trained, including the unchanged native distribution.
    runner.alg.optimizer=torch.optim.Adam(chain(runner.alg.actor.parameters(),runner.alg.critic.parameters()),
                                          lr=runner.alg.learning_rate)
    ids={id(p) for group in runner.alg.optimizer.param_groups for p in group['params']}
    assert all(id(p) in ids for p in actor.parameters())
    obs=env.get_observations()
    reference=Actor(ad.no,ad.na,c['history'],dims['ng'],c['goal_frames'],c['profile']).cuda()
    reference.load_state_dict(ck['modules'][1])
    with torch.no_grad():
        h=torch.randn(8,c['history'],ad.no,device='cuda:0')
        g=torch.randn(8,c['goal_frames'],dims['ng'],device='cuda:0')
        delta=float((policy.mlp(torch.cat((h.flatten(1),g.flatten(1)),-1))-reference(h,g)).abs().max())
    assert delta<1e-6,delta
    (out/'initialization_check.json').write_text(json.dumps(dict(max_actor_output_difference=delta,
        optimizer_includes_encoder_and_decoder=True,shared_actor_and_critic_weights=True,
        shared_normalization=True,initial_action_std=.1)))
    (out/'config.json').write_text(json.dumps(dict(args=vars(args),ppo=cfg,
        actor_architecture='MBRL goal encoder + 10-frame decoder',native_rsl_rl_path=rsl_rl.__file__),indent=2,default=str))
    def export(stage):
        torch.save(dict(format_version=3,args=c,dims=dims,modules=[{},actor.state_dict()],
                   algorithm='native_ppo_matched_MBRL'),out/(stage+'.pt'))
    if args.resume:
        runner.load(args.resume)
        runner.alg.learning_rate=runner.alg.optimizer.param_groups[0]['lr']
        saved=torch.load(args.resume,weights_only=False)
        restored=runner.alg.actor.state_dict()
        delta=max(float((restored[k]-v).abs().max()) for k,v in saved['actor_state_dict'].items())
        assert delta==0
        critic_delta=max(float((runner.alg.critic.state_dict()[k]-v).abs().max()) for k,v in saved['critic_state_dict'].items())
        assert critic_delta==0
        (out/'resume_check.json').write_text(json.dumps(dict(source=args.resume,
            actor_max_difference=delta,critic_max_difference=critic_delta,
            optimizer_state_entries=len(runner.alg.optimizer.state),
            iteration=runner.current_learning_iteration,learning_rate=runner.alg.learning_rate,
            simulator_restarted=True)))
    export('initial')
    initial_encoder=[p.detach().clone() for p in actor.encoder.parameters()]
    clock=time.monotonic();steps=0;rows=[]
    log=(out/'metrics.jsonl').open('w',buffering=1);original_step=env.step
    def timed_step(action):
        nonlocal steps
        if time.monotonic()-clock>=args.seconds:raise Deadline()
        result=original_step(action);steps+=1
        rows.append(float(result[1].mean()))
        if steps%240==0:
            log.write(json.dumps(dict(elapsed=time.monotonic()-clock,sim_steps=steps*args.num_envs,
                    reward_native=sum(rows)/len(rows)))+'\n');rows.clear()
        return result
    env.step=timed_step
    try:
        runner.learn(100000,init_at_random_ep_len=False)
    except Deadline:pass
    runner.save(str(out/'native_final.pt'));export('final')
    change=max(float((x-p).abs().max()) for x,p in zip(initial_encoder,actor.encoder.parameters()))
    (out/'status.json').write_text(json.dumps(dict(status='finished',elapsed=time.monotonic()-clock,
        sim_steps=steps*args.num_envs,iteration=runner.current_learning_iteration,encoder_max_weight_change=change)))
    log.close();ad.env.close()


if __name__=='__main__':main()
