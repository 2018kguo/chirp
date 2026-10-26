"""Top allocation sites at the heap's peak, from a dhat-heap.json.

    python3 scripts/dhat_top.py dhat-heap.json [calls] [--churn]

--churn ranks sites by allocation count over the whole run instead.

Groups program points by their first frame inside a crate of interest and
prints bytes live at the global peak (and per call, if `calls` is given).
"""
import collections
import json
import sys

data = json.load(open(sys.argv[1]))
calls = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 1
churn = "--churn" in sys.argv
frames = data["ftbl"]
interesting = ("chirp::", "tungstenite", "axum", "tokio::sync", "tokio::runtime", "reqwest", "hyper", "rustls", "ort::", "serde_json", "base64")
by_site = collections.Counter()
for point in data["pps"]:
    stack = [frames[i] for i in point["fs"]]
    site = next((f for f in stack if any(k in f for k in interesting)), stack[0] if stack else "?")
    site = site.split(" (")[0].strip()
    by_site[site] += point["tbk"] if churn else point["gb"]
total = sum(by_site.values())
if churn:
    print(f"allocations over the run: {total:,} ({total / calls:,.0f} per call)")
    for site, count in by_site.most_common(25):
        print(f"{count / calls:9.0f} /call  {100 * count / total:5.1f}%  {site[:150]}")
else:
    print(f"live at peak: {total / 1e6:.1f} MB ({total / calls / 1e3:.1f} KB per call over {calls} calls)")
    for site, size in by_site.most_common(25):
        print(f"{size / calls / 1e3:9.1f} KB/call  {100 * size / total:5.1f}%  {site[:150]}")
