"""Parallel native PPO / MBRL comparison; first run or explicit continuation."""
import argparse,subprocess,sys,json,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--motion',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=600)
    p.add_argument('--resume-from',default='')
    p.add_argument('--num-envs',type=int,default=256)
    a=p.parse_args()
    if a.seconds<=0:p.error('--seconds must be positive')
    root=Path(__file__).resolve().parent;out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    motion=Path(a.motion).resolve();base=Path(a.resume_from).resolve() if a.resume_from else None
    m=[sys.executable,str(root/'train_mbrl.py'),'--kind','tracking','--motion',str(motion),
       '--seconds',str(a.seconds),'--num-envs',str(a.num_envs),'--simple-standing','--noise-scale','0',
       '--no-return-sampling','--output',str(out/'mbrl')]
    q=[sys.executable,str(root/'train_ppo.py'),'--motion',str(motion),'--seconds',str(a.seconds),
       '--num-envs',str(a.num_envs),'--output',str(out/'ppo')]
    if base:
        # Fixed normalization is unchanged on continuation; the MBRL checkpoint
        # contains its current weights but the same original normalization buffers.
        m+=['--resume',str(base/'mbrl/final.pt')]
        q+=['--common-init',str(base/'mbrl/final.pt'),'--resume',str(base/'ppo/native_final.pt')]
    else:q+=['--common-init',str(out/'mbrl/initial.pt')]
    manifest=dict(seconds_this_run=a.seconds,resume_from=str(base) if base else None,
                  motion_sha256=hashlib.sha256(motion.read_bytes()).hexdigest(),
                  num_envs=a.num_envs,shared_gpu=True,seed=42)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    commands={'mbrl':m,'ppo':q};logs={k:open(out/(k+'.log'),'w') for k in commands}
    procs={k:subprocess.Popen(cmd,cwd=root,stdout=logs[k],stderr=subprocess.STDOUT) for k,cmd in commands.items()}
    (out/'status.json').write_text(json.dumps({'status':'training','pids':{k:v.pid for k,v in procs.items()}}))
    codes={k:v.wait() for k,v in procs.items()};results={}
    for name in commands:
        logs[name].close();results[name]={'training_exit_code':codes[name]}
        if codes[name]:continue
        for stage in ['initial','final']:
            output=out/name/f'evaluation-{stage}.json'
            with open(out/name/f'evaluation-{stage}.log','w') as log:
                result=subprocess.run([sys.executable,str(root/'evaluate.py'),'--checkpoint',str(out/name/(stage+'.pt')),
                    '--motion',str(motion),'--output',str(output)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            results[name][stage]=json.loads(output.read_text()) if result.returncode==0 else {'error':'evaluation failed'}
    (out/'results.json').write_text(json.dumps(results,indent=2))
    (out/'status.json').write_text(json.dumps({'status':'finished','training_exit_codes':codes}))
    if any(codes.values()):raise SystemExit(1)

if __name__=='__main__':main()
