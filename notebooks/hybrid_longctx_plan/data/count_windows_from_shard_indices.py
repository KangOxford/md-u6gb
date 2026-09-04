import json, os, glob, collections
IDX = os.path.join(os.environ['SP'], 'idx')
TICKERS = "GOOG AAPL NVDA AMZN META TSLA MSFT AMD".split()
Ls = [500, 1000, 2000, 4000]
rows = collections.defaultdict(dict)   # (ticker, month) -> {file: msg_rows, book_rows}
tot_msgs = 0; tot_files = 0
per_month = collections.defaultdict(lambda: [0,0])   # month -> [files, msgs]
win = {L: 0 for L in Ls}
win_files = {L: 0 for L in Ls}
per_tk_msgs = collections.Counter(); per_tk_files = collections.Counter()
per_tk_win2000 = collections.Counter()
per_month_win = {L: collections.Counter() for L in Ls}
daylist = collections.Counter()
for d in sorted(os.listdir(IDX)):
    p = os.path.join(IDX, d, 'index.json')
    if not os.path.exists(p): continue
    j = json.load(open(p))
    shapes = j['shapes']
    # pair message with orderbook
    msg = {}; book = {}
    for k, v in shapes.items():
        tk = k.split('/')[0]
        if tk not in TICKERS: continue
        if '_message_' in k: msg[k.replace('_message_', '_XX_')] = v['shape'][0]
        elif '_orderbook_' in k: book[k.replace('_orderbook_', '_XX_')] = v['shape'][0]
    for k, mrows in msg.items():
        brows = book.get(k)
        if brows is None: continue
        tk = k.split('/')[0]
        tot_files += 1; tot_msgs += mrows
        per_month[d][0] += 1; per_month[d][1] += mrows
        per_tk_msgs[tk] += mrows; per_tk_files[tk] += 1
        daylist[(tk, k.split('_')[1])] += 1
        for L in Ls:
            usable = min(mrows, brows)   # inference=0
            n = max((usable - (L-1)) // L, 0)
            win[L] += n
            if n > 0: win_files[L] += 1
            per_month_win[L][d] += n
            if L == 2000: per_tk_win2000[tk] += n
print(f"files(ticker-days) = {tot_files}")
print(f"total messages     = {tot_msgs:,}")
print(f"total tokens(26)   = {tot_msgs*26:,}")
print()
print(f"{'L(msgs)':>8} {'tokens/win':>11} {'windows':>14} {'files>0':>9} {'0.1% windows':>13}")
for L in Ls:
    print(f"{L:>8} {L*26:>11,} {win[L]:>14,} {win_files[L]:>9} {round(win[L]*0.001):>13,}")
print()
print("per-ticker (files, messages, windows@L2000):")
for tk in TICKERS:
    print(f"  {tk:>5} {per_tk_files[tk]:>6,} {per_tk_msgs[tk]:>14,} {per_tk_win2000[tk]:>10,}")
print()
print("windows@L500 per month (first 6 / last 6):")
ms = sorted(per_month_win[500])
for m in ms[:6] + ['...'] + ms[-6:]:
    print(f"  {m}: {per_month_win[500].get(m,0):,}" if m!='...' else "  ...")
json.dump({'files':tot_files,'messages':tot_msgs,
           'windows':{str(L):win[L] for L in Ls},
           'win_files':{str(L):win_files[L] for L in Ls},
           'per_ticker_files':dict(per_tk_files),'per_ticker_messages':dict(per_tk_msgs),
           'per_ticker_win2000':dict(per_tk_win2000),
           'per_month_win500':dict(per_month_win[500]),
           'per_month_win2000':dict(per_month_win[2000]),
           'per_month_files':{k:v[0] for k,v in per_month.items()},
           'per_month_messages':{k:v[1] for k,v in per_month.items()}},
          open(os.path.join(os.environ['SP'],'dataset_stats_8stock.json'),'w'), indent=1)
