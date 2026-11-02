from pathlib import Path
import json,subprocess
R=Path(__file__).resolve().parents[1];B=Path(json.loads((R/'runtime.json').read_text())['ffmpegDirectory']);F=B/'ffmpeg.exe'
args=[str(F),'-y','-hide_banner','-i',str(R/'renders'/'This-Film-Has-a-Memory-4K.mp4'),'-filter_complex','[0:v]scale=1920:1080:flags=lanczos,split=2[clean][base];[base]ass=captions.ass:fontsdir=assets/fonts/captions[captioned]']
for label,name in [('clean','This-Film-Has-a-Memory-1080p.mp4'),('captioned','This-Film-Has-a-Memory-1080p-Captioned.mp4')]:
 args+=['-map',f'[{label}]','-map','0:a','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',str(R/'renders'/name)]
with (R/'.runtime'/'DELIVERY-ENCODE.log').open('w') as log:subprocess.run(args,cwd=R,stdout=log,stderr=log,check=True)
print('Clean and captioned1080p delivery files created')
