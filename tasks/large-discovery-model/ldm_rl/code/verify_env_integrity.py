"""按安装记录逐包核对文件是否真的在,找出「半装」的包。

为什么需要:2026-09-01 一次 OOM 打断了 pip,留下的 openai 包**只剩子目录、
顶层 .py 全没了**。它在 pip list 里完全正常(版本号都在),import openai 也不报错,
直到踩到 openai._models 才炸 —— 而那已经是 train.py 启动到一半的时候。

pip check 只查依赖关系,查不出文件缺失。可靠的依据是每个包自己的
dist-info/RECORD(安装时写下的文件清单),拿它去比对文件系统。

用法: python verify_env_integrity.py [只查这几个包,逗号分隔]
"""
from __future__ import annotations

import sys
from importlib.metadata import distributions
from pathlib import Path

ONLY = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None

bad = []
checked = 0
for dist in distributions():
    name = dist.metadata["Name"]
    if not name:
        continue
    if ONLY and name not in ONLY:
        continue
    files = dist.files
    if files is None:          # 没有 RECORD(如 -e 装的),跳过并说明
        continue
    missing = []
    for f in files:
        # .pyc 与 RECORD 自身允许缺失
        s = str(f)
        if s.endswith(".pyc") or s.endswith("RECORD") or "__pycache__" in s:
            continue
        try:
            if not (dist.locate_file(f)).exists():
                missing.append(s)
        except Exception:
            missing.append(s)
    checked += 1
    if missing:
        bad.append((name, dist.version, len(files), missing))

print(f"扫了 {checked} 个有安装记录的包")
if not bad:
    print("全部完整:安装记录里的每个文件都在。")
    sys.exit(0)

print(f"\n发现 {len(bad)} 个包的文件与安装记录对不上:\n")
for name, ver, total, missing in sorted(bad, key=lambda x: -len(x[3])):
    print(f"  {name}=={ver}  记录 {total} 个文件,缺 {len(missing)} 个")
    for m in missing[:5]:
        print(f"      缺 {m}")
    if len(missing) > 5:
        print(f"      ... 另有 {len(missing)-5} 个")
print()
print("修法:pip install --no-cache-dir --force-reinstall --no-deps <包>==<版本>")
sys.exit(1)
