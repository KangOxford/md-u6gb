import sys, tarfile, time
sys.path.insert(0, "/lus/lfs1aip2/projects/public/u6gb/tasks/minimax_h3_20260812T/code")
import numpy as np
import h3nano as H
from preprocess_vggsound import decode_clip

nf = H.snap_num_frames(73)
exp = int(round(nf / H.FPS * H.AUDIO_SAMPLING_RATE))
print(f"target: {nf} frames @256px, stereo {exp} samples @32kHz", flush=True)
got, t0 = [], time.time()
tb = "/lus/lfs1aip2/projects/public/u6gb/tasks/minimax_h3_20260812T/data/vggsound/vggsound_01.tar.gz"
with tarfile.open(tb, mode="r|gz") as tar:
    for m in tar:
        if not m.isfile() or not m.name.endswith(".mp4"): continue
        blob = tar.extractfile(m).read(); stem = m.name.rsplit("/",1)[-1][:-4]
        got.append((stem, decode_clip((stem+"|x", blob), nf, 256, H.AUDIO_SAMPLING_RATE)))
        if len(got) >= 6: break
print(f"decoded {len(got)} clips in {time.time()-t0:.1f}s\n", flush=True)
ok = 0
for stem, r in got:
    if r is None: print(f"  {stem:26s} DECODE RETURNED None", flush=True); continue
    _n, v, a = r
    checks = {"vshape": v.shape==(nf,256,256,3), "vdtype": v.dtype==np.uint8,
              "ashape": a.shape==(2,exp), "afinite": bool(np.isfinite(a).all()),
              "vnotblank": float(v.std())>1.0, "anotsilent": float(np.abs(a).max())>1e-4}
    bad=[k for k,val in checks.items() if not val]; ok += not bad
    print(f"  {stem:26s} v{v.shape} std={v.std():5.1f}  a{a.shape} peak={np.abs(a).max():.4f}  {'OK' if not bad else 'FAIL '+','.join(bad)}", flush=True)
print(f"\n{ok}/{len(got)} clips decoded correctly", flush=True)
