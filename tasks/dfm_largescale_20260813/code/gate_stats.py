#!/usr/bin/env python3
"""r2g1 门控残差参数统计。

回答三个问题：
  1. b 从初始 −2.0 学到了哪里（决定默认开度 σ(b)）；
  2. ||w|| 是否离开零初始化（离开 = 门是输入依赖的，没离开 = 门只是常数阻尼）；
  3. 与 P 矩阵（dfm_residual_proj）的谱范数并排，看有效残差幅度 σ(b)·||P|| 相比
     无门格（r2d1）的 ||P|| 是压了还是没压。

登录节点纯 CPU、单文件读，几秒钟完事。
"""
import sys, numpy as onp
from flax import serialization

for path in sys.argv[1:]:
    with open(path, 'rb') as f:
        params = serialization.msgpack_restore(f.read())
    # 参数树可能嵌套（params/...），拍平找目标键
    flat = {}
    def walk(d, pfx=''):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, pfx + k + '/')
            else:
                flat[pfx + k] = onp.asarray(v)
    walk(params)
    tag = path.split('/')[-1].replace('_state.msgpack', '')
    w = next((v for k, v in flat.items() if k.endswith('dfm_res_gate_w')), None)
    b = next((v for k, v in flat.items() if k.endswith('dfm_res_gate_b')), None)
    P = next((v for k, v in flat.items() if k.endswith('dfm_residual_proj')), None)
    print(f'== {tag} ==')
    if w is None:
        print('   (无 gate 参数)')
    else:
        bval = float(b.ravel()[0])
        sig = 1.0 / (1.0 + onp.exp(-bval))
        print(f'   b = {bval:+.4f}  (初始 -2.0)   sigma(b) = {sig:.4f}  (初始 0.1192)')
        print(f'   ||w||_2 = {float(onp.linalg.norm(w)):.4f}  (初始 0)   '
              f'max|w_i| = {float(onp.abs(w).max()):.5f}')
    if P is not None:
        s = onp.linalg.svd(P.astype(onp.float64), compute_uv=False)
        print(f'   ||P||_2(谱) = {float(s[0]):.4f}   ||P||_F = {float(onp.linalg.norm(P)):.4f}')
