#!/usr/bin/env python3
"""ledger_kit -- 从 sgpur 提取的泳道图（GPU-time ledger swimlane）渲染器。

出处（原始实现）
----------------
/projects/public/u6gb/.local/bin/sgpur 是完整工具：采样 sacct/squeue、清洗数据、
渲染整页并发布到 surge.sh。本文件只提取其中「把行/段数据画成泳道图」这一层，
逐块对应关系：

    FILL 色表          <- sgpur:90-96   （满载绿 / 中载黄 / 低载橙 / held 灰 / pending 虚线）
    pct()              <- sgpur:739-741 （时刻 -> track 内百分比定位）
    tick_step()        <- sgpur:749-754 （自动选网格间隔，标签 ~90px 一个）
    tick_times()       <- sgpur:773-790 （网格锚定在整点/午夜，不锚定窗口起点）
    grid_html()        <- sgpur:757-770 （竖直网格线 + 「现在」竖线）
    ruler_html()       <- sgpur:793-813 （共享时间轴，行尾空 tail 保证与数据行同宽）
    seg_div()          <- sgpur:855-872 （一个段：定位/着色/内嵌标签/异常右缘红条）
    CSS                <- sgpur:118-556 的子集，类名加 sgl- 前缀做命名空间隔离

为什么要前缀隔离：sgpur 的 CSS 写在自己独占的页面里，可以直接用 .row/.track；
本 kit 的输出会被嵌进 Jupyter notebook 或别人的页面，裸类名会和宿主样式互相
污染，CSS 变量也从 :root 收缩到 .sgl-root 容器上。

两种输出模式
------------
    render_ledger(..., embed=False)  -> 完整独立 HTML 页面（含 hover 浮层 JS）
    render_ledger(..., embed=True)   -> 单个 <div>，可直接塞进 notebook cell 输出
                                        （tooltip 退化为原生 title 属性，因为
                                        JupyterLab 对未信任 notebook 会剥离 <script>）

数据模型：一行 = 一个实验/作业身份；一段 = 一次连续运行；段间的空洞 = 中断。
时间一律传 naive datetime，调用者自己转好想显示的时区。
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# ---------------------------------------------------------------- 数据模型

@dataclass
class Seg:
    """一次连续运行。kind 决定填充色（沿用 sgpur 的语义）：
    full >=80% 在算 / mid 35-80% / low <35% / held 占着没算 / pending 排队 / pred 预测"""
    start: datetime
    end: datetime
    text: str = ""            # 段内标签（段太窄时自动省略）
    kind: str = "full"
    tip: str = ""             # hover 全文（作业号、命令行等放这里）
    bad: bool = False         # True = 异常结束，右缘画红条（sgpur .seg.bad）


@dataclass
class Row:
    """一行一个实验身份。label/meta/name 对应 sgpur 左列的 jid/规格/名字三件。"""
    label: str
    meta: str = ""
    name: str = ""
    segs: list = field(default_factory=list)
    tail: tuple = ("", "")    # 行尾统计盒左右两格，如 ("3 runs", "评测通过")
    mark: str = ""            # 末段右侧的状态字，如 "→ running" / "TIMEOUT"
    mark_bad: bool = False
    tip: str = ""


# ---------------------------------------------------------------- 视觉常量
# sgpur:90-96 原值照搬；print-friendly 手绘 R6 时间线配色。
FILL = {
    "full":    ("#a6d3a6", "#79b479"),
    "mid":     ("#f2d98a", "#d9b74e"),
    "low":     ("#eeb98c", "#d99356"),
    "held":    ("#eceef0", "#d3d8dd"),
    "pending": ("#f7f8f9", "#dfe3e7"),
    "pred":    ("#fdf9ee", "#b8860b"),   # 预测段：sgpur .seg.pred 的麻布黄
}

CSS = """
.sgl-root{
  --sgl-bg:#fff; --sgl-fg:#16191d; --sgl-dim:#6b7280; --sgl-faint:#9aa1aa;
  --sgl-line:#e6e9ec; --sgl-rule:#f0f2f4; --sgl-accent:#2f6fb5; --sgl-bad:#c0392b;
  --sgl-lw:250px; --sgl-tailw:190px;
  background:var(--sgl-bg); color:var(--sgl-fg);
  font:14px/1.45 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  padding:10px 4px;
}
@media (prefers-color-scheme:dark){
  .sgl-root{
    --sgl-bg:#14171a; --sgl-fg:#e6e9ec; --sgl-dim:#9aa4ae; --sgl-faint:#6b7681;
    --sgl-line:#2a3037; --sgl-rule:#22272c; --sgl-accent:#7fb3e8; --sgl-bad:#e58f8f;
  }
}
.sgl-root *{box-sizing:border-box}
.sgl-h1{font-size:16px; margin:0 0 2px; font-weight:640; letter-spacing:-.01em}
.sgl-sub{color:var(--sgl-dim); font-size:12px; margin-bottom:12px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
/* 一行 = 固定标签列 + 按时间等比的 track + 固定统计盒（sgpur:299-315,416-425） */
.sgl-row{display:flex; align-items:stretch; margin:2px 0}
.sgl-lbl{flex:0 0 var(--sgl-lw); padding-right:12px; overflow:hidden; white-space:nowrap;
  display:flex; align-items:center; gap:7px; font-size:12px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.sgl-lbl .sgl-jid{font-weight:640; font-size:12.5px}
.sgl-lbl .sgl-meta{color:var(--sgl-faint); font-size:11px; flex:0 0 auto}
.sgl-lbl .sgl-nm{color:var(--sgl-dim); overflow:hidden; text-overflow:ellipsis}
.sgl-track{position:relative; flex:1 1 auto; height:26px; min-width:0;
  border-radius:5px; background:var(--sgl-rule); overflow:hidden}
/* 段：4px 圆角、1px 边、窄段藏字（sgpur:346-360） */
.sgl-seg{position:absolute; top:0; bottom:0; border-radius:4px; overflow:hidden;
  border:1px solid; display:flex; align-items:center; cursor:default; min-width:1px}
.sgl-seg span{padding:0 6px; font-size:10.5px; color:#2b3138; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis}
.sgl-seg.sgl-pending{border-style:dashed}
.sgl-seg.sgl-pred{border:1px dashed #b8860b;
  background:repeating-linear-gradient(135deg, transparent 0 4px,
    rgba(184,134,11,.16) 4px 8px)}
.sgl-seg.sgl-bad{box-shadow:inset -3px 0 0 var(--sgl-bad)}
/* 网格线与「现在」线（sgpur:398-401） */
.sgl-gl{position:absolute; top:0; bottom:0; width:1px; background:var(--sgl-line);
  pointer-events:none}
.sgl-gl.sgl-day{background:var(--sgl-dim); opacity:.45}
.sgl-now{position:absolute; top:-2px; bottom:-2px; width:2px; background:var(--sgl-accent);
  pointer-events:none; opacity:.8}
/* 末段状态字（sgpur:402-407） */
.sgl-mark{position:absolute; top:50%; transform:translate(4px,-50%); font-size:10px;
  color:var(--sgl-dim); white-space:nowrap; pointer-events:none}
.sgl-mark.sgl-flip{transform:translate(calc(-100% - 4px),-50%)}
.sgl-mark.sgl-bad{color:var(--sgl-bad)}
/* 共享时间轴（sgpur:409-413） */
.sgl-track.sgl-ruler{background:none; border-radius:0; overflow:visible; height:16px;
  margin-top:5px}
.sgl-axis{position:absolute; left:0; right:0; top:-5px; height:1px; background:var(--sgl-line)}
.sgl-tk{position:absolute; top:0; font-size:10.5px; color:var(--sgl-dim);
  white-space:nowrap; transform:translateX(-50%)}
.sgl-tk.sgl-day{color:var(--sgl-fg); font-weight:600}
/* 行尾统计盒：定宽，保证所有行的 track 右缘对齐（sgpur:416-425） */
.sgl-tail{flex:0 0 var(--sgl-tailw); margin-left:11px; padding:0 9px; overflow:hidden;
  display:flex; align-items:center; justify-content:space-between; gap:8px;
  border:1px solid var(--sgl-line); border-radius:5px;
  font-size:11px; color:var(--sgl-dim); white-space:nowrap}
.sgl-tail.sgl-spacer{border-color:transparent}
.sgl-tail .sgl-g{color:var(--sgl-fg); font-weight:640}
.sgl-tail .sgl-b{color:var(--sgl-bad); font-weight:600}
/* 图例（sgpur:519-522） */
.sgl-legend{display:flex; flex-wrap:wrap; gap:16px; margin-top:14px; padding-top:12px;
  border-top:1px solid var(--sgl-line); font-size:11.5px; color:var(--sgl-dim)}
.sgl-legend i{display:inline-block; width:22px; height:11px; border-radius:3px;
  border:1px solid; margin-right:6px; vertical-align:-1px}
@media (max-width:820px){
  .sgl-lbl{flex-basis:130px; font-size:11px}
  .sgl-seg span{display:none}
  .sgl-tail{display:none}
}
"""

# 页面模式的 hover 浮层（sgpur:679-695 的 tooltip 三事件，逐行同构）。
TIP_JS = """
(function(){
  var tip=document.getElementById('sgl-tip');
  document.addEventListener('mouseover',function(e){
    var t=e.target.closest('[data-tip]'); if(!t) return;
    tip.textContent=t.getAttribute('data-tip'); tip.style.display='block';
  });
  document.addEventListener('mousemove',function(e){
    if(tip.style.display!=='block') return;
    var w=tip.offsetWidth,h=tip.offsetHeight,x=e.clientX+14,y=e.clientY+16;
    if(x+w>innerWidth-8) x=e.clientX-w-14;
    if(y+h>innerHeight-8) y=e.clientY-h-16;
    tip.style.left=Math.max(8,x)+'px'; tip.style.top=Math.max(8,y)+'px';
  });
  document.addEventListener('mouseout',function(e){
    if(e.target.closest('[data-tip]')) tip.style.display='none';
  });
})();
"""

TIP_CSS = """
#sgl-tip{position:fixed; z-index:99; display:none; max-width:640px; pointer-events:none;
  background:#1d2126; color:#f2f4f6; padding:9px 11px; border-radius:6px;
  font:11.5px/1.5 ui-monospace,Menlo,Consolas,monospace; white-space:pre-wrap;
  box-shadow:0 6px 22px rgba(0,0,0,.28); word-break:break-all}
"""


# ---------------------------------------------------------------- 几何函数

def esc(t) -> str:
    return _html.escape(str(t), quote=True)


def att(t) -> str:
    """属性值转义，换行写成 &#10;（sgpur:728-736 的理由：CDN/minifier 会吞字面换行）。"""
    return _html.escape(str(t), quote=True).replace("\n", "&#10;")


def pct(t: datetime, t0: datetime, t1: datetime) -> float:
    span = (t1 - t0).total_seconds() or 1.0
    return max(0.0, min(100.0, (t - t0).total_seconds() / span * 100.0))


def tick_step(span_s: float) -> float:
    """选一个让标签 ~90px 一个的网格间隔（sgpur:749）。"""
    for cand in (300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400,
                 172800, 604800):
        if span_s / cand <= 16:
            return cand
    return 604800


def tick_times(t0: datetime, t1: datetime, step: float) -> list:
    """网格锚定到整点/午夜而非窗口起点：18:00 的线可读，17:43 的线是噪声（sgpur:773）。"""
    base = t0.replace(hour=0, minute=0, second=0, microsecond=0)
    out, k = [], int((t0 - base).total_seconds() // step)
    while True:
        t = base + timedelta(seconds=k * step)
        k += 1
        if t > t1:
            break
        if t >= t0:
            out.append(t)
    return out


def grid_html(t0, t1, step, now=None) -> str:
    out = []
    for t in tick_times(t0, t1, step):
        cls = "sgl-gl sgl-day" if t.hour == 0 and t.minute == 0 else "sgl-gl"
        out.append(f'<i class="{cls}" style="left:{pct(t, t0, t1):.4f}%"></i>')
    if now is not None and t0 <= now <= t1:
        out.append(f'<i class="sgl-now" style="left:{pct(now, t0, t1):.4f}%"></i>')
    return "".join(out)


def ruler_html(t0, t1, step) -> str:
    """共享时间轴。行尾放一个透明 spacer，让轴与数据行严格同宽（sgpur:806-813）。"""
    ticks = []
    for t in tick_times(t0, t1, step):
        midnight = t.hour == 0 and t.minute == 0
        cls = "sgl-tk sgl-day" if midnight else "sgl-tk"
        lab = t.strftime("%m-%d") if midnight else t.strftime("%H:%M")
        if step >= 86400 and not midnight:
            lab = t.strftime("%m-%d %H:%M")
        ticks.append(f'<span class="{cls}" style="left:{pct(t, t0, t1):.4f}%">{lab}</span>')
    return ('<div class="sgl-row"><div class="sgl-lbl"></div>'
            '<div class="sgl-track sgl-ruler"><div class="sgl-axis"></div>'
            + "".join(ticks) +
            '</div><div class="sgl-tail sgl-spacer"></div></div>')


def seg_div(seg: Seg, t0, t1, tooltip_attr: str) -> str:
    """一个段（sgpur:855-872 同构）：定位、着色、窄段藏字、异常右缘红条。"""
    left = pct(seg.start, t0, t1)
    width = max(pct(seg.end, t0, t1) - left, 0.12)
    bg, bd = FILL.get(seg.kind, FILL["held"])
    cls = f"sgl-seg sgl-{seg.kind}" + (" sgl-bad" if seg.bad else "")
    label = f"<span>{esc(seg.text)}</span>" if seg.text and width > 3 else ""
    tip = f' {tooltip_attr}="{att(seg.tip)}"' if seg.tip else ""
    style = f"left:{left:.4f}%;width:{width:.4f}%"
    if seg.kind != "pred":
        style += f";background:{bg};border-color:{bd}"
    return f'<div class="{cls}" style="{style}"{tip}>{label}</div>'


def row_html(row: Row, t0, t1, grid: str, tooltip_attr: str) -> str:
    segs = "".join(seg_div(s, t0, t1, tooltip_attr) for s in row.segs)
    mark = ""
    if row.mark and row.segs:
        x = pct(max(s.end for s in row.segs), t0, t1)
        cls = "sgl-mark sgl-bad" if row.mark_bad else "sgl-mark"
        if x > 82:
            cls += " sgl-flip"
        mark = f'<span class="{cls}" style="left:{x:.4f}%">{esc(row.mark)}</span>'
    tl, tr = (row.tail + ("", ""))[:2]
    tail = (f'<div class="sgl-tail"><span class="sgl-g">{esc(tl)}</span>'
            f'<span>{esc(tr)}</span></div>') if (tl or tr) else \
           '<div class="sgl-tail sgl-spacer"></div>'
    tip = f' {tooltip_attr}="{att(row.tip)}"' if row.tip else ""
    return (f'<div class="sgl-row"{tip}><div class="sgl-lbl">'
            f'<span class="sgl-jid">{esc(row.label)}</span>'
            f'<span class="sgl-meta">{esc(row.meta)}</span>'
            f'<span class="sgl-nm">{esc(row.name)}</span></div>'
            f'<div class="sgl-track">{grid}{segs}{mark}</div>{tail}</div>')


LEGEND_ITEMS = [
    ("full", "运行且在计算"), ("held", "占着 GPU 没在算"),
    ("pending", "排队中"), ("pred", "预测（未发生）"),
]


def legend_html(items=None) -> str:
    parts = []
    for kind, text in (items or LEGEND_ITEMS):
        bg, bd = FILL[kind]
        dash = "border-style:dashed;" if kind in ("pending", "pred") else ""
        parts.append(f'<span><i style="background:{bg};border-color:{bd};{dash}"></i>{esc(text)}</span>')
    parts.append(f'<span><i style="background:#fff;border-color:#d3d8dd;'
                 f'box-shadow:inset -3px 0 0 #c0392b"></i>异常结束（右缘红条）</span>')
    return f'<div class="sgl-legend">{"".join(parts)}</div>'


# ---------------------------------------------------------------- 入口

def render_ledger(title: str, subtitle: str, rows: list,
                  t0: datetime | None = None, t1: datetime | None = None,
                  now: datetime | None = None, embed: bool = False,
                  legend: bool = True, label_width: int = 250) -> str:
    """rows: list[Row]。embed=True 返回可嵌 notebook 的 <div>；否则返回完整页面。"""
    all_times = [t for r in rows for s in r.segs for t in (s.start, s.end)]
    if not all_times:
        raise ValueError("rows 里没有任何段")
    t0 = t0 or min(all_times)
    t1 = t1 or max(all_times)
    pad = timedelta(seconds=(t1 - t0).total_seconds() * 0.02)
    t0w, t1w = t0 - pad, t1 + pad
    step = tick_step((t1w - t0w).total_seconds())
    grid = grid_html(t0w, t1w, step, now)
    tooltip_attr = "title" if embed else "data-tip"

    body = [f'<div class="sgl-root" style="--sgl-lw:{label_width}px">',
            f'<div class="sgl-h1">{esc(title)}</div>' if title else "",
            f'<div class="sgl-sub">{esc(subtitle)}</div>' if subtitle else ""]
    body += [row_html(r, t0w, t1w, grid, tooltip_attr) for r in rows]
    body.append(ruler_html(t0w, t1w, step))
    if legend:
        body.append(legend_html())
    body.append("</div>")
    inner = "\n".join(x for x in body if x)

    if embed:
        return f"<style>{CSS}</style>\n{inner}"
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{CSS}{TIP_CSS}</style></head>"
            f"<body style='margin:0;background:#fff'>"
            f"<div style='max-width:1500px;margin:0 auto;padding:24px 22px 40px'>{inner}</div>"
            f"<div id='sgl-tip'></div><script>{TIP_JS}</script></body></html>")
