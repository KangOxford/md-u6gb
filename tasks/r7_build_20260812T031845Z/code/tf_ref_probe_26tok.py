#!/usr/bin/env python3
"""26tok 的 teacher-forced reference 探针 —— 与 R6/R7 探针同口径的对照。

# 为什么 token 准确率就是 26tok 的 exact-target

26tok 的 L1 用 `(time_s_ref, time_ns_ref)` 在簿里做**精确**匹配。而纳秒时间戳在
可见目标内唯一标识订单（已验证：27,789/27,789 唯一）。所以
「模型把 time_ref 那 5 个 token 全预测对」⟺「L1 精确命中真实目标」= exact-target。

26tok 是**定长** 26 token/消息，字段位置固定（源码注释）：
    evt:0, dir:1, price:2-4, size:5-6, dt_s:7, dt_ns:8-10,
    time_s:11-12, time_ns:13-15, price_ref:16-18, size_ref:19-20, time_ref:21-25
所以不必像变长那样逐字段追踪 span，直接切 token 21-25。

# 口径（与 R6/R7 探针一致）

- 同一批 condition 窗口（250 条真实消息/序列 × 255 序列）。
- 每个位置喂真实前缀（shifted = BOS + toks[:-1]），取该位置 logits 的 argmax。
- 只统计 touch（event_type 2/3/4）消息的 ref 字段。
- 报三个数：time_ref 全 5 token 正确率（= exact-target）、首 token 正确率、
  price_ref/size_ref 正确率（供诊断，26tok 的 resolver 从不用它们查簿）。
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

COND = "/lus/lfs1aip2/projects/public/u6gb/tasks/varlen_bench_subset_20260809/varlen_eval255/data_cond"

# 26tok 定长布局（来自 encoding_26tok.py 的 TOK_LENS 注释）
SPAN_PRICE_REF = (16, 19)
SPAN_SIZE_REF = (19, 21)
SPAN_TIME_REF = (21, 26)      # time_s_ref 21-22 + time_ns_ref 23-25
MSG_LEN = 26
EV_TOUCH = (2, 3, 4)


def main():
    wt = os.environ["W26_WORKTREE"]
    sys.path.insert(0, wt + "/src")
    sys.path.insert(0, wt)
    import jax
    import jax.numpy as jnp
    from lob.encoding import Vocab, Message_Tokenizer, encode_msgs
    from lob.init_train import init_train_state, load_checkpoint, load_metadata
    import lob.validation_helpers as valh
    from s5.registry import build_backbone

    ckpt_root = os.environ["CKPT_ROOT"]
    step = int(os.environ["CKPT_STEP"])
    max_seq = int(os.environ.get("MAX_SEQ", "0"))

    v = Vocab()
    tok = Message_Tokenizer()
    V = len(v)
    args = load_metadata(ckpt_root)
    SEQ = int(os.environ.get("SEQ_LEN", 13000))
    book_dim = int(getattr(args, "book_depth", 500)) + 3
    print(f"[26tok] vocab={V}  SEQ={SEQ}  book_dim={book_dim}", flush=True)

    st0, model_cls = init_train_state(args, n_classes=V, seq_len=SEQ,
                                      book_dim=book_dim, book_seq_len=SEQ)
    ck = load_checkpoint(st0, ckpt_root, step=step, train=False,
                         partial_restore=True)
    state = ck["model"] if isinstance(ck, dict) else ck
    model = model_cls(training=False, step_rescale=1.0)
    bb = build_backbone(args)
    hs = int(args.ssm_size_base)
    if bb.definition.name == "s5" and getattr(args, "conj_sym", False):
        hs //= 2
    bn = bool(getattr(args, "batchnorm", False))

    def mk_hidden():
        return model.initialize_carry(
            1, hidden_size=hs, n_message_layers=args.n_message_layers,
            n_book_pre_layers=args.n_book_pre_layers,
            n_book_post_layers=args.n_book_post_layers,
            n_fused_layers=args.n_layers, h_size_ema=args.d_model,
            d_book=getattr(args, "d_book", 503), **dict(bb.carry_kwargs))

    def _tf_kernel(hidden0, toks, books, valid):
        def _step(h, x):
            token, book_row, val = x

            def _consume(hh):
                hh2, logits = valh.apply_model(
                    hh, jnp.reshape(token, (1,)),
                    jnp.reshape(book_row, (1, -1)),
                    state, model, bn, False)
                return hh2, jnp.argmax(jnp.reshape(logits, (-1,)))

            return jax.lax.cond(val, _consume,
                                lambda hh: (hh, jnp.int32(-1)), h)

        return jax.lax.scan(_step, hidden0, (toks, books, valid))

    tf_compiled = jax.jit(_tf_kernel)

    sys.path.insert(0, wt + "/scripts")
    from importlib import import_module
    gb = import_module("varlen_bench_generate")

    stat = defaultdict(int)
    files = sorted(glob.glob(f"{COND}/*message*.csv"))
    if max_seq:
        files = files[:max_seq]

    for fi, cpath in enumerate(files):
        base = os.path.basename(cpath)
        sid = base.split("_real_id_")[1].split(".csv")[0]
        date = base.split("_")[1]
        try:
            m6, b40, bi = gb.load_cond(COND, sid, date)
        except Exception:
            stat["seq_load_fail"] += 1
            continue

        # LOBSTER 6 列 -> preproc 需要的 DataFrame
        mdf = pd.DataFrame({
            "time": [float(x) for x in m6[:, 0]],
            "event_type": m6[:, 1].astype(int),
            "order_id": m6[:, 2].astype(int),
            "size": m6[:, 3].astype(int),
            "price": m6[:, 4].astype(int),
            "direction": m6[:, 5].astype(int),
        })
        bdf = pd.DataFrame(b40)
        try:
            m_ = tok.preproc(mdf, bdf)
        except Exception as exc:
            stat["preproc_fail"] += 1
            if stat["preproc_fail"] <= 2:
                print(f"[26tok] preproc 失败: {type(exc).__name__}: {exc}", flush=True)
            continue

        m_ = np.asarray(m_)
        try:
            toks = np.asarray(encode_msgs(m_, v.ENCODING)).reshape(-1)
        except Exception as exc:
            stat["encode_fail"] += 1
            if stat["encode_fail"] <= 2:
                print(f"[26tok] encode 失败: {type(exc).__name__}: {exc}", flush=True)
            continue

        n_msg = len(toks) // MSG_LEN
        cond_book, _mid = gb._condition_book_features(
            m6, b40, bi, price_levels=int(getattr(args, "book_depth", 500)))
        # 定长：第 i 条消息的所有 token 都看第 i 条的 book
        msg_idx = np.repeat(np.arange(n_msg), MSG_LEN)
        nb = min(len(cond_book), n_msg)
        msg_idx = np.clip(msg_idx, 0, nb - 1)
        target_books = cond_book[msg_idx]

        shifted = np.concatenate(
            [np.asarray([0], dtype=np.int32), toks[:-1].astype(np.int32)])
        n = min(len(shifted), SEQ)
        pt = np.zeros(SEQ, dtype=np.int32)
        pb = np.zeros((SEQ, book_dim), dtype=np.float32)
        pv = np.zeros(SEQ, dtype=bool)
        pt[-n:] = shifted[-n:]
        pb[-n:] = target_books[-n:]
        pv[-n:] = True

        _h, preds = tf_compiled(mk_hidden(), jnp.asarray(pt),
                                jnp.asarray(pb), jnp.asarray(pv))
        preds = np.asarray(preds)[-n:]
        toks_al = toks[-n:]
        offset = len(toks) - n

        et_col = m_[:, 1].astype(int) if m_.ndim == 2 else None
        for mi in range(n_msg):
            if et_col is None or mi >= len(et_col):
                break
            if int(et_col[mi]) not in EV_TOUCH:
                continue
            base_i = mi * MSG_LEN - offset
            if base_i < 0 or base_i + MSG_LEN > len(preds):
                continue
            stat["touch_scored"] += 1
            for name, (a, b) in (("time_ref", SPAN_TIME_REF),
                                 ("price_ref", SPAN_PRICE_REF),
                                 ("size_ref", SPAN_SIZE_REF)):
                t_true = toks_al[base_i + a: base_i + b]
                t_pred = preds[base_i + a: base_i + b]
                if len(t_true) and np.array_equal(t_pred, t_true):
                    stat[name + "_exact"] += 1
                if len(t_true) and t_pred[0] == t_true[0]:
                    stat[name + "_top1"] += 1

        if (fi + 1) % 10 == 0:
            print(f"[26tok] {fi+1}/{len(files)} 序列  已评 {stat['touch_scored']} touch",
                  flush=True)

    n = max(stat["touch_scored"], 1)
    out = {
        "sequences": len(files),
        "touch_scored": stat["touch_scored"],
        "time_ref_exact_pct": 100.0 * stat["time_ref_exact"] / n,
        "time_ref_top1_pct": 100.0 * stat["time_ref_top1"] / n,
        "price_ref_exact_pct": 100.0 * stat["price_ref_exact"] / n,
        "size_ref_exact_pct": 100.0 * stat["size_ref_exact"] / n,
        "fails": {k: stat[k] for k in
                  ("seq_load_fail", "preproc_fail", "encode_fail")},
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
