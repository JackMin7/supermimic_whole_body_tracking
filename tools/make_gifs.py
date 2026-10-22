"""Convert paired simulation MP4s to normal-speed, looping GIFs (requires ffmpeg)."""
import subprocess
from pathlib import Path
media=Path(__file__).resolve().parents[1]/'media'
for seconds in (60,):
    source=media/f'ppo_vs_mbrl_{seconds}s.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-filter_complex',
        'fps=10,scale=768:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer',
        '-loop','0',str(media/'PPO与MBRL站立对照_60秒动图.gif')],check=True)
