#!/usr/bin/env python3
"""Download a public Google Drive file by ID (handles virus-scan confirm page)."""
import re, sys, urllib.request, urllib.parse, os

def download(file_id, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print(f"SKIP (exists): {dest} ({os.path.getsize(dest)} bytes)"); return True
    base = "https://drive.usercontent.google.com/download"
    url = base + "?" + urllib.parse.urlencode({"id": file_id, "export": "download"})
    for attempt in range(3):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        head = data[:3000].decode("utf-8", "replace")
        if "Virus scan warning" in head or "uc-download-link" in head:
            conf = re.search(r'name="confirm" value="([^"]*)"', head)
            uuid = re.search(r'name="uuid" value="([^"]*)"', head)
            if not conf or not uuid:
                print(f"FAIL (confirm page unparsed): {file_id}"); return False
            url = base + "?" + urllib.parse.urlencode({
                "id": file_id, "export": "download",
                "confirm": conf.group(1), "uuid": uuid.group(1)})
            continue
        if data[:2] == b"PK" or head.lstrip().startswith("<!DOCTYPE html>") is False or len(data) > 100000:
            with open(dest, "wb") as f: f.write(data)
            print(f"OK: {dest} ({len(data)} bytes)")
            return True
        print(f"FAIL (unexpected content): {file_id}"); return False
    print(f"FAIL (retries exhausted): {file_id}"); return False

if __name__ == "__main__":
    fid, dest = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    sys.exit(0 if download(fid, dest) else 1)
