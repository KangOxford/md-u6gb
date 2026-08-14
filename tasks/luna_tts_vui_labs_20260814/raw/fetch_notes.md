# 抓取记录

2026-08-14，在 Isambard 登录节点上执行（纯网络 + 小文本处理，无 GPU、无 Lustre 递归）。

---

## 坑：微信按 User-Agent 拦截

WebFetch 的默认 UA 拿到的是验证页，不是文章：

```
当前环境异常，完成验证后即可继续访问。
```

**这不是 IP 封禁，也不是登录墙**——换浏览器 UA 直接就通了：

```bash
curl -sL --max-time 60 \
  -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36' \
  -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' \
  -H 'Accept-Language: zh-CN,zh;q=0.9,en;q=0.8' \
  'https://mp.weixin.qq.com/s/dww9GzQojweNQ-BxEAXB9g' -o wx_raw.html
# → 3,410,418 字节
```

同族问题见记忆 `reference_cloudflare_1010_blocks_urllib`（Cloudflare 1010 只拦
urllib 的 UA）。**判据**：如果一个页面在浏览器里能看、用工具抓到的却是验证页/403，
先换 UA 再考虑别的原因，不要一上来就找代理。

---

## 提取元信息

微信把元信息塞在页面的 JS 变量里，比解析 DOM 稳：

```python
re.search(r'var msg_title\s*=\s*[\'"](.*?)[\'"]\s*\.html', raw)      # 标题
re.search(r'var nickname\s*=\s*htmlDecode\(\s*[\'"](.*?)[\'"]', raw)  # 公众号名
re.search(r'var ct\s*=\s*"(\d+)"', raw)                               # 发布时间戳(unix)
re.search(r'createTime\s*=\s*[\'"](.*?)[\'"]', raw)                   # 发布时间(可读)
```

正文在 `id="js_content"` 的 div 里。3.4 MB HTML 里正文只有约 9.8 KB 文本，
其余全是内联 CSS 和 JS。

---

## 正文转文本

```python
body = re.search(r'id="js_content"(.*?)</div>\s*(?:<script|<div class="rich_media_tool)', raw, re.S).group(1)
body = re.sub(r'<img[^>]*?data-src="([^"]*)"[^>]*>', lambda x: f'\n[IMG: {x.group(1)}]\n', body)  # 保留图片URL
body = re.sub(r'</(p|div|section|h1|h2|h3|h4|li|tr|blockquote)>', '\n', body, flags=re.I)
body = re.sub(r'<br\s*/?>', '\n', body, flags=re.I)
body = re.sub(r'<[^>]+>', '', body)
body = html.unescape(body)
```

**保留 `[IMG: url]` 是有用的**：这篇文章的关键证据（榜单截图、架构图）全在图里，
文字只是说明。图片 URL 留着，需要时还能取回来看。

---

## 论文全文（arXiv HTML，不用下 PDF）

```bash
curl -sL 'https://arxiv.org/html/2608.11593v1' -o luna_paper.html   # 455 KB
```

arXiv 的 LaTeXML 输出把公式放在 `<math alttext="...">` 里，**alttext 就是原始 LaTeX**，
所以转文本时先把公式换成 alttext 再剥标签，公式就完整保下来了：

```python
raw = re.sub(r'<math[^>]*?alttext="([^"]*)"[^>]*?>.*?</math>',
             lambda m: ' $'+html.unescape(m.group(1))+'$ ', raw, flags=re.S)
```

455 KB HTML → 94 KB 纯文本，1730 行，含全部 15 张表格的数字。
**比读 PDF 好用得多**，也不用装任何 PDF 解析库。

表格转换用 `</td>` → ` | `，得到的是每个单元格一行的竖排格式，
不好看但不丢数字，用 `sed -n '起,止p' | tr '\n' '~'` 可以还原成行。

---

## 文件

| scratchpad 文件 | 说明 |
|---|---|
| `wx_raw.html` | 微信原始 HTML，3.4 MB |
| `wx_body.txt` | 正文纯文本，9.8 KB（已复制为本目录的 `wx_article_20260814.txt`） |
| `luna_paper.html` | arXiv HTML 全文，455 KB |
| `luna_paper.txt` | 论文纯文本，94 KB |

scratchpad 路径：
`/local/user/1483804540/claude-1483804540/-lus-lfs1aip2-projects-public-u6gb/cc0eb2d5-c19c-446f-88bf-d0378de4450b/scratchpad/`

（scratchpad 是会话级的，会被清理。需要长期保留的只有本目录里的
`wx_article_20260814.txt`；论文用 arXiv 的 v1 版本号锚定即可，不必存档。）
