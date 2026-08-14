import sys, torch
T = "/lus/lfs1aip2/projects/public/u6gb/tasks/minimax_h3_20260812T"
b = torch.load(f"{T}/data/latents/latents_0000.pt", map_location="cpu", weights_only=False)
v, a, l = b["video"].float(), b["audio"].float(), b["label"]
print(f"video {tuple(v.shape)}  mean {v.mean():+.4f}  std {v.std():.4f}")
print(f"audio {tuple(a.shape)}  mean {a.mean():+.4f}  std {a.std():.4f}")
# per-channel, which is the axis the statistics were meant for
vc = v.permute(1,0,2,3,4).reshape(v.shape[1], -1)
ac = a.permute(2,0,1,3).reshape(a.shape[2], -1)
print(f"video per-channel mean range [{vc.mean(1).min():+.3f}, {vc.mean(1).max():+.3f}]  "
      f"std range [{vc.std(1).min():.3f}, {vc.std(1).max():.3f}]")
print(f"audio per-channel mean range [{ac.mean(1).min():+.3f}, {ac.mean(1).max():+.3f}]  "
      f"std range [{ac.std(1).min():.3f}, {ac.std(1).max():.3f}]")
print(f"finite: video {bool(torch.isfinite(v).all())}  audio {bool(torch.isfinite(a).all())}")
print(f"distinct labels: {len(set(l.tolist()))} of 310")
print()
print("Normalization is per latent channel, so the per-channel means should sit near 0")
print("and the per-channel stds near 1. A wrong axis leaves them scattered -- and E6")
print("cannot see it, because the round trip never normalizes.")
