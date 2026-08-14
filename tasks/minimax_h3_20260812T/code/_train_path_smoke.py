"""Exercise the training data path on real shards before the corpus finishes.

`LatentCorpus` reads `manifest.json`, which the encoder only writes at the very end,
so without this the first test of the training path would come 85 minutes from now --
and any shape or dtype mismatch would surface then rather than while there is time to
fix it cheaply. This builds the corpus from whatever shards exist and runs one real
forward/backward through H3-nano.
"""
import json, sys, time, torch
T = "/lus/lfs1aip2/projects/public/u6gb/tasks/minimax_h3_20260812T"
sys.path.insert(0, f"{T}/code")
import h3nano as H

shards = sorted(__import__("pathlib").Path(f"{T}/data/latents").glob("latents_*.pt"))
print(f"shards on disk: {[s.name for s in shards]}", flush=True)
vids, auds, labs = [], [], []
for s in shards:
    b = torch.load(s, map_location="cpu", weights_only=False)
    vids.append(b["video"]); auds.append(b["audio"]); labs.append(b["label"])
video = torch.cat(vids); audio = torch.cat(auds); label = torch.cat(labs).long()
bank = torch.load(f"{T}/data/latents/text_bank.pt", map_location="cpu", weights_only=False)
text = bank["embeds"]
print(f"corpus  video {tuple(video.shape)}  audio {tuple(audio.shape)}  text {tuple(text.shape)}", flush=True)
print(f"labels  {label.min().item()}..{label.max().item()}  distinct {len(set(label.tolist()))}", flush=True)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
layout = H.build_layout(text.shape[1], video.shape[2], video.shape[3], video.shape[4],
                        audio.shape[3]).to(dev)
print(f"layout  seq={layout.sequence_length}  video={len(layout.video_indices)} "
      f"text={len(layout.text_indices)} audio={len(layout.audio_indices)}", flush=True)

model = H.build_transformer(H.NANO_CONFIG).to(dev)
model.enable_gradient_checkpointing()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
census = H.parameter_census(model)
print(f"model   {census['TOTAL']/1e6:.3f} M params", flush=True)

g = torch.Generator().manual_seed(0)
t0 = time.time()
for step in range(3):
    idx = torch.randint(len(video), (8,), generator=g)
    v = video[idx].to(dev, torch.float32); a = audio[idx].to(dev, torch.float32)
    tx = text[label[idx]].to(dev, torch.float32)
    batch = H.make_flow_batch(v, a, tx, layout)
    vp, ap = model(**batch.transformer_kwargs())
    loss, logs = H.flow_loss(vp, ap, batch)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    print(f"  step {step}  loss {logs['loss']:.4f} (v {logs['loss_video']:.4f} "
          f"a {logs['loss_audio']:.4f})  t_v {logs['t_video']:.3f} t_a {logs['t_audio']:.3f}", flush=True)
print(f"\n3 real training steps in {time.time()-t0:.1f}s -- training path works on real data")
print(f"peak GPU mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB" if dev.type=="cuda" else "")
