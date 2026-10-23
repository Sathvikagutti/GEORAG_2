#!/usr/bin/env python3
"""List contents of a public Google Drive folder (no auth) by folder ID."""
import re, json, sys, urllib.request

def fetch(folder_id):
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")

def parse(html):
    m = re.search(r"window\['_DRIVE_ivd'\] = '((?:[^'\\]|\\.)*)'", html, re.S)
    if not m:
        return []
    raw = m.group(1)
    out = []; i = 0
    simple = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"', "/": "/"}
    while i < len(raw):
        if raw[i] == "\\" and i + 1 < len(raw):
            c = raw[i + 1]
            if c == "x" and i + 3 < len(raw):
                out.append(chr(int(raw[i + 2:i + 4], 16))); i += 4; continue
            if c in simple:
                out.append(simple[c]); i += 2; continue
            out.append(raw[i]); i += 1; continue
        out.append(raw[i]); i += 1
    decoded = "".join(out)
    dec = json.JSONDecoder(); idx = 0; objs = []
    while idx < len(decoded):
        while idx < len(decoded) and decoded[idx] in " \n\r\t": idx += 1
        if idx >= len(decoded): break
        try:
            o, off = dec.raw_decode(decoded, idx); objs.append(o); idx = off
        except Exception:
            break
    items = []
    def walk(node):
        if isinstance(node, list) and node and isinstance(node[0], list):
            for c in node: walk(c)
        elif isinstance(node, list) and len(node) > 3 and isinstance(node[0], str):
            items.append({
                "id": node[0], "name": node[2], "mime": node[3],
                "modified": node[8] if len(node) > 8 and isinstance(node[8], (int, float)) else None,
            })
    for e in objs: walk(e)
    return items

def fmt_size(n):
    if n is None: return ""
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"

if __name__ == "__main__":
    fid = sys.argv[1]
    html = fetch(fid)
    items = parse(html)
    print(f"# folder {fid}: {len(items)} items")
    for it in sorted(items, key=lambda x: (x["mime"] != "application/vnd.google-apps.folder", x["name"])):
        kind = "DIR " if it["mime"] == "application/vnd.google-apps.folder" else "FILE"
        print(f"{kind}\t{it['name']}\t{it['mime']}\t{it['id']}")
