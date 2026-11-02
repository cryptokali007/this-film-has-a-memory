from pathlib import Path
import re
R=Path(__file__).resolve().parents[1]
text=(R/'captions.vtt').read_text(encoding='utf-8')
cues=re.findall(r'(\d\d:\d\d:\d\d\.\d{3}) --> (\d\d:\d\d:\d\d\.\d{3})\n([^\n]+)',text)
assert len(cues)>0
def ass_time(s):return s[:-4]+'.'+s[-3:-1]
header='''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,40,&H00EDF3ED,&H00EDF3ED,&H00141007,&H80141007,0,0,0,0,100,100,0,0,3,11,0,2,100,100,78,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
events='\n'.join(f'Dialogue: 0,{ass_time(a)},{ass_time(b)},Default,,0,0,0,,{s}' for a,b,s in cues)
(R/'captions.ass').write_text(header+events+'\n',encoding='utf-8')
(R/'captions.srt').write_text('\n\n'.join(f'{i+1}\n{a.replace(".",",")} --> {b.replace(".",",")}\n{s}' for i,(a,b,s) in enumerate(cues))+'\n',encoding='utf-8')
print(f'{len(cues)} canonical phrase cues converted to SRT/ASS')
