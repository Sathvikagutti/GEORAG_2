#!/usr/bin/env python3
"""Build the final shareable project package:

  1. INDEX.md        - top-level index (hand-written structure + generated file list)
  2. MANIFEST.csv    - path,size,sha256 of every packaged file
  3. GeoRAG_Member2_ModelUpgrade.zip - the verified ZIP

Excludes: .venv, logs, __pycache__, extracted/ (regenerable by unzipping
drive_downloads/), derived/*.npy (regenerable by src/01_build_features.py),
and any hidden files. Verifies the ZIP afterwards by re-listing it and comparing
against the manifest.
"""
import hashlib
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_NAME = "GeoRAG_Member2_ModelUpgrade.zip"
EXCLUDE_DIRS = {".venv", "logs", "__pycache__", "extracted", ".git"}
EXCLUDE_FILES = {ZIP_NAME, "MANIFEST.csv"}
EXCLUDE_SUFFIXES = (".pyc",)
# derived npy arrays are regenerable; keep the manifest json only
DERIVED_KEEP = {"manifest.json"}


def should_skip(path):
    rel = os.path.relpath(path, ROOT)
    parts = rel.split(os.sep)
    for p in parts:
        if p in EXCLUDE_DIRS or p.startswith("."):
            return True
    base = parts[-1]
    if base in EXCLUDE_FILES or base.endswith(EXCLUDE_SUFFIXES):
        return True
    if parts[0] == "derived" and base not in DERIVED_KEEP:
        return True
    return False


def iter_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS
                       and not d.startswith(".")]
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            if not should_skip(p):
                yield p


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest():
    rows = ["path,size_bytes,sha256"]
    files = sorted(iter_files())
    for p in files:
        rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
        rows.append(f"{rel},{os.path.getsize(p)},{sha256(p)}")
    out = os.path.join(ROOT, "MANIFEST.csv")
    with open(out, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    return [r.split(",")[0] for r in rows[1:]], out


def build_zip(expected):
    zpath = os.path.join(ROOT, ZIP_NAME)
    if os.path.exists(zpath):
        os.remove(zpath)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in expected:
            z.write(os.path.join(ROOT, rel), "GeoRAG_Member2_ModelUpgrade/" + rel)
        # ship the manifest inside the package (a manifest cannot contain its own hash,
        # so it is written after hashing and added here)
        z.write(os.path.join(ROOT, "MANIFEST.csv"),
                "GeoRAG_Member2_ModelUpgrade/MANIFEST.csv")
    return zpath


def verify_zip(zpath, expected):
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        if bad is not None:
            return False, [f"CRC failure: {bad}"]
        names = [n for n in z.namelist() if not n.endswith("/")]
    stripped = sorted(n.replace("GeoRAG_Member2_ModelUpgrade/", "", 1) for n in names)
    problems = []
    if stripped != sorted(expected):
        e, g = set(expected), set(stripped)
        problems += [f"missing from zip: {m}" for m in sorted(e - g)]
        problems += [f"unexpected in zip: {u}" for u in sorted(g - e)]
    return not problems, problems


def main():
    expected, mpath = write_manifest()
    print(f"manifest: {mpath} ({len(expected)} files)")
    zpath = build_zip(expected)
    print(f"zip: {zpath} ({os.path.getsize(zpath):,} bytes)")
    ok, problems = verify_zip(zpath, expected + ["MANIFEST.csv"])
    print("ZIP VERIFIED" if ok else "ZIP VERIFY FAILED")
    for p in problems:
        print("  ", p)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
