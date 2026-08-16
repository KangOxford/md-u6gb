# ledger_kit — sgpur 泳道图渲染器（提取封装版）

> 速览：把 sgpur 网页里那种「一行一实验、块是运行、洞是中断」的泳道图
> 变成一个可复用的纯 stdlib Python 模块。两种输出：独立 HTML 页面，或
> 可直接嵌进 Jupyter notebook cell 的 `<div>`。

## 出处（考古结论）

| 问题 | 答案 |
|---|---|
| 命令在哪 | `/projects/public/u6gb/.local/bin/sgpur`（Python 单文件，3421 行；`sgpu` 是同目录的终端版数据层） |
| 生成的 HTML 在哪 | `~/.cache/sgpur/site/`（`SGPUR_SITE`，sgpur:55-58），发布到 `https://gpu-ledger-q7x2m4.surge.sh/` |
| 绘图逻辑在哪 | CSS sgpur:118-556；几何函数 `pct/tick_step/tick_times/grid_html/ruler_html` sgpur:739-813；段渲染 `seg_div` sgpur:855-872；行结构 `overview_section` sgpur:901-931、`lanes_section` sgpur:1091- |

本 kit 的 `ledger.py` 逐块标注了与 sgpur 的行号对应关系；改动只有两类：
类名加 `sgl-` 前缀 + CSS 变量收缩到容器（嵌入宿主页面/notebook 时不互相污染），
以及 embed 模式下 tooltip 从 JS 浮层退化为原生 `title`（JupyterLab 对未信任
notebook 剥离 `<script>`）。配色、几何、网格锚定规则与 sgpur 完全一致。

## 文件

| 文件 | 作用 |
|---|---|
| `ledger.py` | 渲染器本体（`Row`/`Seg` 数据模型 + `render_ledger()`），零依赖 |
| `example_xvla.py` | 用例：Hyper-XVLA 三个月实验全史（真实 job id 与日期） |
| `example.html` | 上者产物，浏览器直接打开 |

## 用法

```python
import sys; sys.path.insert(0, "/projects/public/u6gb/sgpu/ledger_kit")
from datetime import datetime as D
from ledger import Row, Seg, render_ledger

rows = [
    Row("6023162", "4N", "hyper-lora-200k",
        segs=[Seg(D(2026,8,15,17,6), D(2026,8,16,17,5), "run-2", "full",
                  tip="hover 全文放这里", bad=False)],
        tail=("~8%", "→ 200K"), mark="→ running"),
]

# ① 独立页面（hover 用 JS 浮层）
open("out.html", "w").write(render_ledger("标题", "副题", rows))

# ② notebook cell 内嵌（hover 用原生 title）
from IPython.display import HTML
HTML(render_ledger("标题", "副题", rows, embed=True))
```

`Seg.kind` 的语义沿用 sgpur：`full`（≥80% 在算，绿）/ `mid`（35–80%，黄）/
`low`（<35%，橙）/ `held`（占着没算，灰）/ `pending`（排队，虚线）/
`pred`（预测段，麻布黄虚线）；`bad=True` 在段右缘画红条表示异常结束。
时间轴自动选网格间隔并锚定到整点/午夜（不是窗口起点），`now=` 画「现在」竖线。
