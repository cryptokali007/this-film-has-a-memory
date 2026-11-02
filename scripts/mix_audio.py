from pathlib import Path
import json,re,subprocess
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'assets'/'audio'; SR=48000; N=300*SR
FF=Path(json.loads((ROOT/'runtime.json').read_text())['ffmpegDirectory'])/'ffmpeg.exe'
score,sr=sf.read(A/'score.wav',always_2d=True,dtype='float64')
assert sr==SR and score.shape==(N,2)
manifest=json.loads((A/'voice-manifest.json').read_text())
voice=np.zeros(N,dtype=np.float64);duck=np.ones(N,dtype=np.float64)
for item in manifest['clips']:
    clip,rate=sf.read(A/item['file'],dtype='float64')
    if rate!=SR:clip=resample_poly(clip,SR//np.gcd(SR,rate),rate//np.gcd(SR,rate))
    rms=np.sqrt(np.mean(clip**2));clip*=min(10**(-21/20)/rms,.82/np.max(np.abs(clip)))
    at=round(item['start_seconds']*SR);end=at+len(clip)
    assert end<N
    voice[at:end]+=clip
    lo=max(0,at-round(.4*SR));hi=min(N,end+round(.8*SR))
    tt=np.arange(lo,hi)/SR;start=at/SR;stop=end/SR
    attack=np.clip((tt-(start-.4))/.4,0,1);release=np.clip((stop+.8-tt)/.8,0,1)
    shape=np.minimum(attack*attack*(3-2*attack),release*release*(3-2*release))
    duck[lo:hi]=np.minimum(duck[lo:hi],1-.82*shape)
sf.write(A/'narration.wav',voice,SR,subtype='PCM_24')
raw=score*(.44*duck[:,None])+voice[:,None]
sf.write(A/'mix-raw.wav',raw,SR,subtype='FLOAT')
cmd=[str(FF),'-hide_banner','-i',str(A/'mix-raw.wav'),'-af','loudnorm=I=-16:TP=-1.5:LRA=9:print_format=json','-f','null','-']
p=subprocess.run(cmd,capture_output=True,text=True,check=True)
m=json.loads(re.findall(r'\{[^{}]+\}',p.stderr)[-1])
f=f"loudnorm=I=-16:TP=-1.5:LRA=9:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true:print_format=json"
p=subprocess.run([str(FF),'-y','-hide_banner','-i',str(A/'mix-raw.wav'),'-af',f,'-ar',str(SR),'-c:a','pcm_s24le',str(A/'final-mix.wav')],capture_output=True,text=True,check=True)
final,sr=sf.read(A/'final-mix.wav',always_2d=True)
assert final.shape==(N,2) and np.isfinite(final).all() and np.max(np.abs(final))<.99
report={'duration':300,'sample_rate':sr,'channels':2,'voice_clips':len(manifest['clips']),'voice_end':max(c['end_seconds'] for c in manifest['clips']),'first_pass':m,'final_pass':json.loads(re.findall(r'\{[^{}]+\}',p.stderr)[-1]),'peak_dbfs':float(20*np.log10(np.max(np.abs(final)))),'ducking':'smooth0.4s attack/0.8s release,18percent music during voice','narration_resampling':'scipy.signal.resample_poly 24kHz to48kHz'}
(ROOT/'evidence'/'MIX_REPORT.json').write_text(json.dumps(report,indent=2))
features=json.loads((A/'score-features.json').read_text());motion={'feature_rate':features['feature_rate_hz'],'normalization':{},'source':'Actual original score, RMS bands normalized to95th percentile'}
for k in ['low','mid','high']:
    arr=np.asarray(features[k+'_rms']);norm=float(np.percentile(arr,95))
    motion['normalization'][k]=norm;motion[k]=np.clip(arr/max(norm,1e-8),0,1.3).round(5).tolist()
(A/'motion-features.json').write_text(json.dumps(motion,separators=(',',':')))
print(json.dumps(report,indent=2))
