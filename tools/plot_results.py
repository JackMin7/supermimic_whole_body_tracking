"""Rebuild Chinese figures from archived logs; no interpolated evaluations."""
import json,csv,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
root=Path(__file__).resolve().parents[1];results=root/'results/standing_1h';media=root/'media'
for path in [os.environ.get('MBRL_CJK_FONT','/nonexistent'),'/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/System/Library/Fonts/PingFang.ttc']:
    if Path(path).exists():
        font_manager.fontManager.addfont(path);plt.rcParams['font.family']=font_manager.FontProperties(fname=path).get_name();break
plt.rcParams['axes.unicode_minus']=False
colors={'ppo':'#3267ac','mbrl':'#d87828'};labels={'ppo':'PPO（原版 BeyondMimic）','mbrl':'MBRL（SuperMimic）'}
rows=[];fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
for name in colors:
    records=[]
    for part,offset in [('initial_10min',0),('continued_50min',600)]:
        records += [dict(json.loads(line),offset=offset) for line in (results/f'{name}_{part}.jsonl').read_text().splitlines()]
    time=np.array([m['elapsed']+m['offset'] for m in records]);reward=np.array([m['reward_native'] for m in records])
    means=[];near=[]
    for minute in range(10,61,10):
        # Allow tiny final-budget overshoot only in the final bin of each run.
        mask=(time>(minute-10)*60)&(time<=minute*60)
        window=(time>(minute-1)*60)&(time<=minute*60)
        value=float(reward[mask].mean());end=float(reward[window].mean());means.append(value);near.append(end)
        selected=time[mask]
        rows.append(dict(method=name,minute=minute,interval_start_minute=minute-10,log_samples=int(mask.sum()),first_log_s=float(selected.min()),last_log_s=float(selected.max()),mean_logged_reward=value,last_minute_mean_logged_reward=end))
    for ax,values in zip(axs,[means,near]):
        ax.plot(range(10,61,10),values,'o-',color=colors[name],label=labels[name])
        for x,y in zip(range(10,61,10),values):ax.annotate(f'{y:.3f}',(x,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
for ax,title in zip(axs,['各十分钟区间的日志奖励均值','各时间点前一分钟的日志奖励均值']):
    ax.set(title=title,xlabel='累计训练时间（分钟）',ylabel='原生任务奖励 / 控制步',xticks=list(range(10,61,10)));ax.legend();ax.grid(alpha=.2);ax.margins(y=.2)
fig.suptitle('PPO 与 MBRL：每十分钟训练指标对照\n日志记录等权聚合；两算法原始平均窗口不同，不代表独立评测',fontsize=13)
fig.savefig(media/'每十分钟训练奖励对照图.png',dpi=180)
with open(results/'ten_minute_training.csv','w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
r=json.loads((results/'evaluation.json').read_text());fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained')
for ax,key,title in zip(axs,['mean_time_before_failure_s','survivors','joint_rmse_before_failure_rad'],['平均维持时间（秒，20 秒封顶）','完成 20 秒的环境数（共 32 个）','失败前关节均方根误差（rad）']):
 for j,name in enumerate(colors):
    vals=[r[name][s][key] for s in ['at_10min','at_60min']]
    if key.startswith('mean_time'):vals=[min(20,v) for v in vals]
    bars=ax.bar(np.arange(2)+(j-.5)*.34,vals,.32,color=colors[name],label=labels[name]);ax.bar_label(bars,fmt='%.3f' if key!='survivors' else '%d',padding=3,fontsize=9)
 ax.set_xticks([0,1],['训练 10 分钟','训练 60 分钟']);ax.set_title(title,fontsize=11);ax.margins(y=.2);ax.grid(axis='y',alpha=.2)
axs[0].legend(fontsize=9);fig.suptitle('已有检查点的独立 20 秒评测\n20 / 30 / 40 / 50 分钟缺少配对检查点，未补画或插值',fontsize=13)
fig.savefig(media/'站立成功数与关节误差对照图.png',dpi=180)
