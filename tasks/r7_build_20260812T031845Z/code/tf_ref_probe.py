#!/usr/bin/env python3
"""Teacher-forced reference 探针：模型对 `ref` 字段的预测有多准，R6 vs R7 resolver 差多少。

# 为什么需要它

生成序列上**没有** exact-target 的 ground truth：被引用的订单是模型自己编出来的，
CSV 里没有一列写着"模型本意指哪一单"。所以此前所有生成侧数字都只是 any-live-ID
（找到了某笔活单）或解析失败率，不是准确率。

teacher-forced 通道解决这个问题：喂**真实**前缀，读模型对**真实**下一条 touch 的
`ref` 预测，与 CSV 里的真实 order_id 比。这是唯一有真值的 exact-target。

# 口径

- 只用 condition 窗口（250 条真实消息/序列）。它的 book features 在生成脚本里已经
  算好，不必重放撮合；窗口内约一半消息是 touch，255 序列足够统计。
- 每个 token 位置喂真实前缀（shifted = BOS + toks[:-1]），取该位置 logits 的 argmax。
- `ref` 字段的 token span 由 tokenizer 逐字段解码追踪得到（字段顺序
  typedir → dt → [t_sec] → price → size → ref）。
- 判定分三层：
    ref_tok_top1   ref 字段首 token 的 argmax 是否等于真实 token
    ref_exact      ref 字段**全部** token 的 argmax 是否逐个等于真实（等价预测对 ref_n）
    resolver_hit   把预测出的 ref_n 交给 R6 / R7 resolver，解析结果是否 == 真实 order_id
- 真实簿状态由 condition 消息逐条重放维护（纯 Python 字典，不需要撮合引擎）。
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import defaultdict
from decimal import Decimal

import numpy as np

COND = "/lus/lfs1aip2/projects/public/u6gb/tasks/varlen_bench_subset_20260809/varlen_eval255/data_cond"
EV_TOUCH = (2, 3, 4)


def decode_spans(tk, tokens, pos):
    """复刻 decode_event 的字段顺序，额外返回每个字段的 [start, end)。"""
    spans = {}
    start = pos
    et, direction = tk.decode_typedir(tk._token_at(tokens, pos))
    pos += 1
    spans["typedir"] = (start, pos)

    p0 = pos
    _dt, used = tk.decode_dt(tokens, pos)
    pos += used
    spans["dt"] = (p0, pos)

    if pos < len(tokens) and tk._is_t_sec_token(tk._token_at(tokens, pos)):
        p0 = pos
        _ts, used = tk.decode_time("t_sec", tokens, pos)
        pos += used
        spans["t_sec"] = (p0, pos)

    p0 = pos
    _pr, used = tk.decode_value("price", tokens, pos)
    pos += used
    spans["price"] = (p0, pos)

    ref_n = None
    if et in (1, 2, 3, 4, 5, 6):
        p0 = pos
        _sz, used = tk.decode_value("size", tokens, pos)
        pos += used
        spans["size"] = (p0, pos)
        if et in (2, 3, 4):
            p0 = pos
            ref_n, used = tk.decode_value("ref", tokens, pos)
            pos += used
            spans["ref"] = (p0, pos)

    return et, direction, ref_n, spans, pos - start


def load_cond_msgs(path):
    """读一个序列的 condition 消息：(time_ns, et, oid, qty, price, side)。"""
    out = []
    for r in csv.reader(open(path, newline="")):
        if len(r) < 6:
            continue
        out.append((int(Decimal(r[0]) * Decimal(10**9)), int(r[1]), r[2],
                    int(r[3]), int(r[4]), int(r[5])))
    return out


def main():
    sys.path.insert(0, os.environ["VARLEN_WORKTREE"] + "/src")
    sys.path.insert(0, os.environ["VARLEN_WORKTREE"])
    import jax
    import jax.numpy as jnp
    from lob import encoding_varlen_R6 as ev
    from lob.init_train import init_train_state, load_checkpoint, load_metadata
    from lob.varlen_resolver_R7 import (COL_OID, COL_PRICE, COL_QTY, COL_TNS,
                                        COL_TS, resolve_R7, _live_at_level,
                                        _rank_order)
    import lob.validation_helpers as valh
    from s5.registry import build_backbone

    ckpt_root = os.environ["CKPT_ROOT"]
    step = int(os.environ["CKPT_STEP"])
    max_seq = int(os.environ.get("MAX_SEQ", "0"))

    tk = ev._tokenizer()
    V = int(tk.layout.total_size)
    args = load_metadata(ckpt_root)
    SEQ = int(os.environ.get("VARLEN_SEQ_LEN", 13000))
    book_dim = int(getattr(args, "book_depth", 500)) + 3

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

    # teacher-forced scan：与 _prefill_kernel 同一条路径，只是把 logits 接出来
    def _tf_kernel(hidden0, toks, books, valid):
        def _step(h, x):
            token, book_row, v = x

            def _consume(hh):
                hh2, logits = valh.apply_model(
                    hh, jnp.reshape(token, (1,)),
                    jnp.reshape(book_row, (1, -1)),
                    state, model, bn, False)
                return hh2, jnp.argmax(jnp.reshape(logits, (-1,)))

            def _skip(hh):
                return hh, jnp.int32(-1)

            return jax.lax.cond(v, _consume, _skip, h)

        return jax.lax.scan(_step, hidden0, (toks, books, valid))

    tf_compiled = jax.jit(_tf_kernel)
    print("[tf] 模型与 tokenizer 就绪", flush=True)

    from importlib import import_module
    gen_mod_path = os.environ["VARLEN_WORKTREE"] + "/scripts"
    sys.path.insert(0, gen_mod_path)
    gb = import_module("varlen_bench_generate")

    stat = defaultdict(int)
    err_hist = defaultdict(int)

    files = sorted(glob.glob(f"{COND}/*message*.csv"))
    if max_seq:
        files = files[:max_seq]

    for fi, cpath in enumerate(files):
        base = os.path.basename(cpath)
        sid = base.split("_real_id_")[1].split(".csv")[0]
        date = base.split("_")[1]
        try:
            m6, b40, bi = gb.load_cond(COND, sid, date)
        except Exception as exc:
            stat["seq_load_fail"] += 1
            continue

        fields = gb.cond_to_lobster14(m6, b40)
        toks, _, _ = tk.encode_messages(**fields)
        toks = np.asarray(toks, dtype=np.int32)
        cond_book, _mid = gb._condition_book_features(
            m6, b40, bi, price_levels=int(getattr(args, "book_depth", 500)))
        msg_idx = np.asarray(ev.message_index_R6(toks), dtype=np.int64)
        target_books = cond_book[msg_idx]

        shifted = np.concatenate(
            [np.asarray([int(tk.BOS_ID)], dtype=np.int32), toks[:-1]])
        n = min(len(shifted), SEQ)
        pt = np.zeros(SEQ, dtype=np.int32)
        pb = np.zeros((SEQ, book_dim), dtype=np.float32)
        pv = np.zeros(SEQ, dtype=bool)
        pt[-n:] = shifted[-n:]
        pb[-n:] = target_books[-n:]
        pv[-n:] = True

        _h, preds = tf_compiled(mk_hidden(), jnp.asarray(pt),
                                jnp.asarray(pb), jnp.asarray(pv))
        preds = np.asarray(preds)[-n:]          # 对齐回 toks 的最后 n 个位置
        toks_al = toks[-n:]

        # 逐条 decode 真实 token，拿 ref 的 span；同时重放簿
        msgs = load_cond_msgs(cpath)
        active = {}
        pos = 0
        mi = 0
        offset = len(toks) - n                  # toks 被截断时的位移
        while pos < len(toks) and mi < len(msgs):
            try:
                et, direction, ref_n, spans, used = decode_spans(tk, toks, pos)
            except Exception:
                stat["decode_fail"] += 1
                break
            now_ns, et_csv, oid, qty, price, side = msgs[mi]

            FIELD = os.environ.get("PROBE_FIELD", "ref")
            if et_csv in EV_TOUCH and FIELD in spans and oid in active:
                s, e = spans[FIELD]
                s2, e2 = s - offset, e - offset
                if 0 <= s2 and e2 <= len(preds):
                    stat["touch_scored"] += 1
                    true_ref_toks = toks_al[s2:e2]
                    pred_ref_toks = preds[s2:e2]
                    if pred_ref_toks[0] == true_ref_toks[0]:
                        stat["ref_tok_top1"] += 1
                    exact = bool(np.array_equal(pred_ref_toks, true_ref_toks))
                    if exact:
                        stat["ref_exact"] += 1

                    # 预测的 ref_n（token 全对时等于真值；否则尝试解码）
                    pred_ref_n = None
                    if exact:
                        pred_ref_n = ref_n
                    else:
                        try:
                            pred_ref_n, _u = tk.decode_value(
                                "ref", list(int(t) for t in pred_ref_toks), 0)
                        except Exception:
                            stat["pred_ref_undecodable"] += 1

                    # 用真实簿解析：R6（纯排名）vs R7（加 quantity 约束）
                    tgt = active[oid]
                    rows, ids = [], []
                    for k2, o in active.items():
                        if o[0] == tgt[0] and o[1] == tgt[1]:
                            rows.append([o[1], o[3], len(ids), 0,
                                         o[4] // 10**9, o[4] % 10**9])
                            ids.append(k2)
                    arr = np.asarray(rows, dtype=np.int64)
                    if pred_ref_n is not None and int(pred_ref_n) > 0:
                        k = int(pred_ref_n)
                        live = _live_at_level(arr, tgt[1])
                        if len(live) >= k >= 1:
                            r6_oid = ids[int(live[_rank_order(live)[-k], COL_OID])]
                        else:
                            r6_oid = None
                        r7_idx, _prov = resolve_R7(arr, tgt[1], k, qty, et_csv)
                        r7_oid = ids[r7_idx] if r7_idx is not None else None
                        stat["r6_hit" if r6_oid == oid else
                             ("r6_miss" if r6_oid is None else "r6_wrong")] += 1
                        stat["r7_hit" if r7_oid == oid else
                             ("r7_miss" if r7_oid is None else "r7_wrong")] += 1
                        if ref_n is not None:
                            err_hist[int(pred_ref_n) - int(ref_n)] += 1
                    else:
                        stat["pred_ref_unusable"] += 1

            # 推进真实簿
            if et_csv == 1:
                active[oid] = [side, price, qty, qty, now_ns]
            elif et_csv in EV_TOUCH and oid in active:
                if et_csv in (2, 4):
                    active[oid][3] -= qty
                    if active[oid][3] <= 0:
                        active.pop(oid, None)
                else:
                    active.pop(oid, None)
            pos += used
            mi += 1

        if (fi + 1) % 10 == 0:
            print(f"[tf] {fi+1}/{len(files)} 序列  已评 {stat['touch_scored']} touch",
                  flush=True)

    n = max(stat["touch_scored"], 1)
    out = {
        "sequences": len(files),
        "touch_scored": stat["touch_scored"],
        "ref_tok_top1_pct": 100.0 * stat["ref_tok_top1"] / n,
        "ref_exact_pct": 100.0 * stat["ref_exact"] / n,
        "r6": {k[3:]: stat[k] for k in ("r6_hit", "r6_wrong", "r6_miss")},
        "r7": {k[3:]: stat[k] for k in ("r7_hit", "r7_wrong", "r7_miss")},
        "misc": {k: stat[k] for k in
                 ("pred_ref_undecodable", "pred_ref_unusable",
                  "decode_fail", "seq_load_fail")},
        "ref_n_error_hist": {str(k): v for k, v in
                             sorted(err_hist.items())[:40]},
    }
    den = sum(out["r6"].values()) or 1
    out["r6_hit_pct"] = 100.0 * out["r6"]["hit"] / den
    out["r7_hit_pct"] = 100.0 * out["r7"]["hit"] / den
    out["r6_wrong_pct"] = 100.0 * out["r6"]["wrong"] / den
    out["r7_wrong_pct"] = 100.0 * out["r7"]["wrong"] / den
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
