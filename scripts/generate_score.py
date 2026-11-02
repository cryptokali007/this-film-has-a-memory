from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import scipy.signal as sig
import soundfile as sf

SR, SECONDS = 48000, 300
BLOCK_SECONDS = 5
BLOCK = SR * BLOCK_SECONDS
ROOT = Path(__file__).resolve().parents[1] / "assets" / "audio"
TMP, OUT = ROOT / "score_float_work.wav", ROOT / "score.wav"

SECTIONS = [
 {"start":0.,"end":24.,"name":"mysterious ocean archive intro","gain":.42,"bpm":72,"kick_every":0,"arp_beats":3.,"arp_gain":.34},
 {"start":24.,"end":56.,"name":"gentle pulse in recall","gain":.55,"bpm":82,"kick_every":4,"arp_beats":1.,"arp_gain":.62},
 {"start":56.,"end":93.,"name":"broad expanding storage world","gain":.72,"bpm":88,"kick_every":4,"arp_beats":1.,"arp_gain":.78},
 {"start":93.,"end":135.,"name":"precise kinetic lattice","gain":.78,"bpm":98,"kick_every":2,"arp_beats":.5,"arp_gain":.92},
 {"start":135.,"end":168.,"name":"suspended certification","gain":.42,"bpm":72,"kick_every":8,"arp_beats":2.,"arp_gain":.48},
 {"start":168.,"end":203.,"name":"intimate encrypted space","gain":.48,"bpm":82,"kick_every":4,"arp_beats":1.,"arp_gain":.50},
 {"start":203.,"end":240.,"name":"strongest rhythmic ribbon studio","gain":.93,"bpm":104,"kick_every":2,"arp_beats":.5,"arp_gain":1.},
 {"start":240.,"end":270.,"name":"warm memory proof","gain":.66,"bpm":84,"kick_every":4,"arp_beats":1.,"arp_gain":.72},
 {"start":270.,"end":293.,"name":"final wide hopeful crescendo","gain":.90,"bpm":88,"kick_every":2,"arp_beats":.5,"arp_gain":.88},
 {"start":293.,"end":300.,"name":"intentional musical tail and fade","gain":.50,"bpm":88,"kick_every":0,"arp_beats":2.,"arp_gain":.14},
]
CHORD_DATA = [
 (0,"Dm9",[38,45,50,53,57,64]),(9,"Bbmaj7",[34,41,46,50,53,57,62]),(17,"Cadd9",[36,43,48,50,55,60,64]),
 (24,"Dm9",[38,45,50,53,57,64]),(32,"Fmaj9",[41,48,53,57,60,64,67]),(40,"Cadd9",[36,43,48,50,55,60,64]),
 (48,"Gm7",[43,50,55,58,62,65]),(56,"Fmaj9",[41,48,53,57,60,64,67]),(66,"Cadd9",[36,43,48,50,55,60,64]),
 (78,"Bbmaj7",[34,41,46,50,53,57,62]),(87,"Fmaj9",[41,48,53,57,60,64,67]),
 (93,"Dm9",[38,45,50,53,57,64]),(101,"Cadd9",[36,43,48,50,55,60,64]),(110,"Gm7",[43,50,55,58,62,65]),
 (119,"Asus4",[45,52,57,59,62,64]),(127,"Dm9",[38,45,50,53,57,64]),
 (135,"Gm7",[43,50,55,58,62,65]),(146,"Bbmaj7",[34,41,46,50,53,57,62]),(157,"Asus4",[45,52,57,59,62,64]),
 (168,"Dm9",[38,45,50,53,57,64]),(178,"Bbmaj7",[34,41,46,50,53,57,62]),(190,"Gm7",[43,50,55,58,62,65]),(199,"Dm/F",[41,45,50,53,57,62]),
 (203,"Fmaj9",[41,48,53,57,60,64,67]),(212,"Cadd9",[36,43,48,50,55,60,64]),(221,"Dm9",[38,45,50,53,57,64]),(230,"Bbmaj7",[34,41,46,50,53,57,62]),
 (240,"Fmaj9",[41,48,53,57,60,64,67]),(249,"Dm9",[38,45,50,53,57,64]),(259,"Bbmaj7/F",[41,46,50,53,57,62]),(266,"Cadd9",[36,43,48,50,55,60,64]),
 (270,"Fmaj9",[41,48,53,57,60,64,67]),(278,"Bbmaj7",[34,41,46,50,53,57,62]),(285,"Fmaj9",[41,48,53,57,60,64,67]),(293,"F6add9",[41,48,53,55,57,60,62,64]),
]
CHORDS = [{"start":float(t),"end":float(CHORD_DATA[i+1][0] if i+1<len(CHORD_DATA) else SECONDS),
           "name":name,"notes":notes} for i,(t,name,notes) in enumerate(CHORD_DATA)]

def hz(m): return 440.0 * 2.0 ** ((float(m)-69.0)/12.0)

def section_for(t):
    for s in SECTIONS:
        if s["start"] <= t < s["end"]: return s
    return SECTIONS[-1]

def section_gain(t):
    out=np.empty_like(t)
    for s in SECTIONS:
        out[(t>=s["start"]) & (t<s["end"])]=s["gain"]
    for i in range(1,len(SECTIONS)):
        b=SECTIONS[i]["start"]; w=1.35; mask=(t>=b-w/2)&(t<b+w/2)
        x=np.clip((t[mask]-(b-w/2))/w,0,1); mix=.5-.5*np.cos(np.pi*x)
        out[mask]=SECTIONS[i-1]["gain"]*(1-mix)+SECTIONS[i]["gain"]*mix
    return out

def fade_out(t):
    x=np.clip((t-292.8)/7.0,0,1)
    return np.where(t>=299.8,0,np.where(t>292.8,np.cos(.5*np.pi*x),1.0))

def make_events():
    kicks=[]; arps=[]; eid=0
    kick_strength={24:.16,56:.22,93:.26,135:.09,168:.12,203:.42,240:.17,270:.25}
    for si,s in enumerate(SECTIONS[:-1]):
        beat=60/s["bpm"]
        if s["kick_every"]:
            k=0
            while s["start"]+k*beat < s["end"]-.1:
                if k % s["kick_every"] == 0:
                    strength=kick_strength.get(s["start"],0)
                    if strength: kicks.append((s["start"]+k*beat+.02,strength,k,si))
                k+=1
        step=max(.22,beat*s["arp_beats"]); n=0; offset=.35 if si==0 else 0.
        t=s["start"]+offset+n*step
        while t<min(s["end"],293):
            if si==0 and n not in (3,8,13,18,23,28):
                n+=1; t=s["start"]+offset+n*step; continue
            if si in (4,5,7) and n%8 in (3,7):
                n+=1; t=s["start"]+offset+n*step; continue
            ci=max(i for i,c in enumerate(CHORDS) if c["start"]<=t); notes=CHORDS[ci]["notes"]
            pat=(0,2,1,3,2,1,4,2); degree=pat[(n+si*2)%8]%len(notes[2:])
            midi=notes[2+degree]+12
            if si in (2,3,6,8) and n%8 in (2,6): midi+=12
            vel=s["arp_gain"]*(.78 if n%16 in (7,15) else 1.)
            arps.append((t,hz(midi),vel,eid,si)); eid+=1
            n+=1; t=s["start"]+offset+n*step
    arps.extend([(5.4,hz(62),.27,eid+1,0),(12.7,hz(65),.28,eid+2,0),(19.6,hz(69),.30,eid+3,0)])
    return sorted(kicks),sorted(arps)

KICKS, ARPS=make_events()
KT=np.array([e[0] for e in KICKS]); AT=np.array([e[0] for e in ARPS])

def add_pad(y,t,t0):
    for c in CHORDS:
        if c["end"]<=t0 or c["start"]>=t0+len(t)/SR: continue
        mask=(t>=c["start"])&(t<c["end"])
        if not np.any(mask): continue
        tt=t[mask]; u=tt-c["start"]; rem=c["end"]-tt
        env=np.minimum(np.sin(.5*np.pi*np.clip(u/2.4,0,1)),np.sin(.5*np.pi*np.clip(rem/2.8,0,1)))
        env*=.92+.08*np.sin(2*np.pi*.11*tt+c["start"]*.17)
        for j,m in enumerate(c["notes"]):
            f=hz(m); amp=.108*(.72 if f<100 else (.88 if f<220 else .62))
            dl=.0015+.00016*j; pl=2*np.pi*f*(1-dl)*tt; pr=2*np.pi*f*(1+dl)*tt
            l=np.sin(pl)+.24*np.sin(2*pl+.08)+.075*np.sin(3*pl+.19)
            r=np.sin(pr+.13)+.24*np.sin(2*pr+.21)+.075*np.sin(3*pr+.29)
            y[mask,0]+=amp*env*l; y[mask,1]+=amp*env*r

def add_arp(y,t,t0,e):
    onset,f,vel,eid,si=e; dur=.95 if si in (2,3,6,8) else 1.15
    i=max(0,int(math.floor((onset-t0)*SR))); j=min(len(t),int(math.ceil((onset+dur-t0)*SR)))
    if i>=j:return
    u=t[i:j]+t0-onset; env=(1-np.exp(-u*100))*np.exp(-u*(4.7 if si in (3,6,8) else 3.9))
    phase=2*np.pi*f*u
    tone=np.sin(phase)+.29*np.sin(2.01*phase+.21)+.095*np.sin(3.91*phase+.42)
    body=np.sin(2*np.pi*f*.503*u+.25)*np.exp(-u*8)
    pan=.42*math.sin(onset*.19+eid*.61); amp=.080*vel
    y[i:j,0]+=amp*env*(tone+.16*body)*.72*(1-pan)
    y[i:j,1]+=amp*env*(tone+.16*body)*.72*(1+pan)

def add_kick(y,t,t0,e):
    onset,strength,beat_no,si=e; i=max(0,int((onset-t0)*SR)); j=min(len(t),int(math.ceil((onset+.34-t0)*SR)))
    if i>=j:return
    u=t[i:j]+t0-onset
    phase=2*np.pi*(44*u+3*(1-np.exp(-20*u)))
    env=np.exp(-9*u)*np.sin(np.minimum(1,u/.012)*np.pi/2)
    kick=.31*strength*env*np.sin(phase)
    knock=.045*strength*np.exp(-85*u)*np.sin(np.minimum(1,u/.003)*np.pi/2)*np.sin(2*np.pi*184*u)
    y[i:j,0]+=kick+knock; y[i:j,1]+=kick+knock

def add_risers(y,t,noise):
    boundaries=sorted(set([s["start"] for s in SECTIONS[1:]]+[56.,93.,203.,240.,270.]))
    for b in boundaries:
        u=t-(b-3.6); mask=(u>=0)&(u<3.6)
        if not np.any(mask):continue
        x=np.clip(u[mask]/3.6,0,1); env=(x*x*(3-2*x))**1.5
        local=u[mask]
        phase=2*np.pi*(90*local+680*3.6*(x**2.4)/2.4)
        val=.012*env*np.sin(phase)+.010*env*noise[mask]
        y[mask,0]+=val; y[mask,1]+=.94*val

def add_final_bloom(y,t):
    x=np.clip((t-270)/23,0,1); env=(x**.75)*((t>=270)&(t<293))
    for m,a,p in ((65,.032,0.),(69,.027,.9),(72,.024,1.8),(76,.014,2.7)):
        f=hz(m); trem=.88+.12*np.sin(2*np.pi*(.14+.01*x)*t+p)
        y[:,0]+=env*a*trem*np.sin(2*np.pi*f*.9994*t+p)
        y[:,1]+=env*a*trem*np.sin(2*np.pi*f*1.0006*t+p+.22)

def add_reflections(dry,history):
    ext=np.concatenate((history,dry)); n=len(dry); base=len(history); y=dry.copy()
    y[:,0]+=.15*ext[base-int(.23*SR):base-int(.23*SR)+n,0]
    y[:,1]+=.14*ext[base-int(.29*SR):base-int(.29*SR)+n,1]
    y[:,1]+=.085*ext[base-int(.41*SR):base-int(.41*SR)+n,0]
    y[:,0]+=.075*ext[base-int(.47*SR):base-int(.47*SR)+n,1]
    y[:,0]+=.052*ext[base-int(.63*SR):base-int(.63*SR)+n,0]
    y[:,1]+=.048*ext[base-int(.71*SR):base-int(.71*SR)+n,1]
    return y,ext[-len(history):].copy()

def synthesize():
    rng=np.random.default_rng(917031)
    sos=sig.butter(2,[360,6500],btype="bandpass",fs=SR,output="sos")
    state_l=np.zeros((len(sos),2)); state_r=np.zeros((len(sos),2))
    history=np.zeros((int(.72*SR)+1,2))
    ot=np.arange(-2,SECONDS+3,.8); ov=sig.savgol_filter(rng.normal(size=len(ot)),9,2,mode="interp")
    with sf.SoundFile(TMP,mode="w",samplerate=SR,channels=2,subtype="FLOAT",format="WAV") as out:
        for start in range(0,SECONDS*SR,BLOCK):
            n=min(BLOCK,SECONDS*SR-start); t0=start/SR; t=t0+np.arange(n,dtype=np.float64)/SR
            y=np.zeros((n,2)); add_pad(y,t,t0)
            ocean=np.interp(t,ot,ov)
            nl, state_l=sig.sosfilt(sos,rng.standard_normal(n),zi=state_l)
            nr, state_r=sig.sosfilt(sos,rng.standard_normal(n),zi=state_r)
            y[:,0]+=.006*ocean+.0019*nl; y[:,1]+=.00558*ocean+.0019*nr
            a=np.searchsorted(AT,t0-1.2); b=np.searchsorted(AT,t0+n/SR)
            for e in ARPS[a:b]: add_arp(y,t,t0,e)
            a=np.searchsorted(KT,t0-.4); b=np.searchsorted(KT,t0+n/SR)
            for e in KICKS[a:b]: add_kick(y,t,t0,e)
            add_final_bloom(y,t)
            add_risers(y,t,.5*(nl+nr))
            y*=section_gain(t)[:,None]
            y,history=add_reflections(y,history)
            # Apply the master tail after reflections so it reaches exact silence by 299.8s.
            y*=fade_out(t)[:,None]
            out.write(y.astype(np.float32))

def render_pcm24():
    peak=0.; sums=np.zeros(2); count=0
    with sf.SoundFile(TMP) as f:
        while True:
            x=f.read(BLOCK,dtype="float64",always_2d=True)
            if len(x)==0:break
            peak=max(peak,float(np.max(np.abs(x)))); sums+=x.sum(0); count+=len(x)
    mean=sums/count; gain=(10**(-3.2/20))/peak
    with sf.SoundFile(TMP) as src, sf.SoundFile(OUT,mode="w",samplerate=SR,channels=2,subtype="PCM_24",format="WAV") as dst:
        while True:
            x=src.read(BLOCK,dtype="float64",always_2d=True)
            if len(x)==0:break
            frame_start=dst.tell()
            times=(frame_start+np.arange(len(x),dtype=np.float64))/SR
            dst.write(((x-mean)*gain*fade_out(times)[:,None]).astype(np.float32))
    return peak,mean,gain

def analyze():
    info=sf.info(OUT)
    if (info.samplerate,info.channels,info.frames,info.subtype)!=(SR,2,SECONDS*SR,"PCM_24"):
        raise RuntimeError(f"Unexpected WAV format: {info}")
    nper=SR//25; win=np.hanning(nper); freq=np.fft.rfftfreq(nper,1/SR)
    masks=((freq>=20)&(freq<160),(freq>=160)&(freq<2500),(freq>=2500)&(freq<=12000))
    ones=np.full(len(freq),2.); ones[0]=1.; ones[-1]=1.
    scale=ones/(nper*np.sum(win*win))
    allr=[]; bands=[[],[],[]]; sections={s["name"]:[] for s in SECTIONS}
    peak=0.; dc=np.zeros(2); total=0
    with sf.SoundFile(OUT) as f:
        while True:
            x=f.read(BLOCK,dtype="float64",always_2d=True)
            if len(x)==0:break
            peak=max(peak,float(np.max(np.abs(x)))); dc+=x.sum(0); total+=len(x)
            mono=x.mean(1); frames=mono.reshape(-1,nper); r=np.sqrt(np.mean(frames*frames,axis=1))
            spec=np.fft.rfft(frames*win[None,:],axis=1); power=(np.abs(spec)**2)*scale[None,:]
            allr.extend(r.tolist())
            for q,m in enumerate(masks):bands[q].extend(np.sqrt(np.sum(power[:,m],axis=1)).tolist())
            base=len(allr)-len(r)
            for i,v in enumerate(r):
                s=section_for((base+i+.5)/25); sections[s["name"]].append(float(v*v))
    dc/=total
    if peak>10**(-3/20)+1e-6 or peak>=1:raise RuntimeError(f"Peak limit/clipping failure: {peak}")
    if max(abs(dc[0]),abs(dc[1]))>.0002:raise RuntimeError(f"DC offset: {dc}")
    srms={k:float(np.sqrt(np.mean(v))) if v else 0. for k,v in sections.items()}
    tail=float(np.max(allr[-5:]))
    if tail>1e-5:raise RuntimeError(f"Tail not silent: {tail}")
    if any(srms[s["name"]]<.002 for s in SECTIONS[:9]):raise RuntimeError("A section is not audibly active")
    fobj={"source":"score.wav decoded after PCM_24 encoding","sample_rate_hz":SR,"channels":2,
          "duration_seconds":SECONDS,"feature_rate_hz":25,"window_samples":nper,
          "band_ranges_hz":{"low":[20,160],"mid":[160,2500],"high":[2500,12000]},
          "times_seconds":[round((i+.5)/25,3) for i in range(len(allr))],
          "rms":[round(v,8) for v in allr],
          "low_rms":[round(v,8) for v in bands[0]],
          "mid_rms":[round(v,8) for v in bands[1]],
          "high_rms":[round(v,8) for v in bands[2]],
          "section_rms":{k:round(v,8) for k,v in srms.items()}}
    (ROOT/"score-features.json").write_text(json.dumps(fobj,separators=(",",":")),encoding="utf-8")
    return info,peak,dc,srms,float(np.sqrt(np.mean(np.square(allr)))),tail

def report(float_peak,mean,gain,result):
    info,peak,dc,srms,overall,tail=result
    lines=["# Film score delivery","",
      "Original procedural stereo instrumental score synthesized in Python from oscillators, shaped noise, and event-driven percussion. No sample packs, existing songs, hosted generators, or API music were used.","",
      "## Form and harmony","",
      "The score opens in D minor with a quiet, water-like filtered noise bed and widely spaced resonant tones. A restrained pulse and recurring mallet phrase enter during recall, then the chord voicings widen into F-major colors as the storage world opens. The 93–135 second lattice tightens the tempo and mallet subdivisions; certification at 135 seconds removes weight and leaves more air. The encrypted-space section returns to a close, low register. The 203–240 second ribbon-studio passage has the strongest beat and most active phrase. Warm memory proof relaxes the rhythm, then F-major strings and rising harmony carry the final crescendo from 270 to 293 seconds. A held F6/add9 sonority rings into a seven-second intentional fade, reaching silence by 299.8 seconds.","",
      "The middle register stays restrained around narration, while bass pulses, short mallet decays, and stereo reflections provide motion. This is an unducked music stem; dialogue mixing remains a parent-stage operation.","",
      "## Signal checks","",
      f"- WAV: {info.channels} channels, {info.samplerate} Hz, {info.subtype}, {info.frames} frames ({info.duration:.3f} seconds).",
      f"- Float render peak before final gain: {float_peak:.6f}; pre-gain DC estimate: {mean.tolist()}.",
      f"- Applied uniform gain: {gain:.6f}. Encoded peak: {peak:.6f} ({20*math.log10(peak):.2f} dBFS).",
      f"- Encoded per-channel DC: {dc.tolist()}; no clipping; final 0.2 s RMS maximum: {tail:.8f}.",
      f"- Overall RMS across 25 Hz feature windows: {overall:.6f}. Features were decoded from the finished PCM_24 WAV.","","## Section activity",""]
    for s in SECTIONS:lines.append(f"- {s['start']:g}–{s['end']:g}s, {s['name']}: RMS {srms[s['name']]:.5f}")
    lines.extend(["","The feature file contains 7,500 values at 25 Hz for full-band RMS and 20–160 Hz, 160–2,500 Hz, and 2,500–12,000 Hz bands.",""])
    (ROOT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")

def main():
    synthesize()
    fp,mean,gain=render_pcm24()
    result=analyze()
    report(fp,mean,gain,result)
    TMP.unlink(missing_ok=True)
    info,peak,dc,srms,overall,tail=result
    print(json.dumps({"wav":str(OUT),"features":str(ROOT/"score-features.json"),
      "report":str(ROOT/"REPORT.md"),"frames":info.frames,"duration":info.duration,
      "peak_dbfs":round(20*math.log10(peak),3),"dc":dc.tolist(),
      "overall_rms":round(overall,6),"tail_rms_max":round(tail,10),
      "section_rms":{k:round(v,5) for k,v in srms.items()}},indent=2))

if __name__=="__main__":main()

