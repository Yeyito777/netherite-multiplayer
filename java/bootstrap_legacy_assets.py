#!/usr/bin/env python3
"""Populate ForgeGradle 2.x's asset cache through Mojang's HTTPS endpoint.

ForgeGradle 2.2 hard-codes the retired HTTP resources URL, which now returns
HTTP 400. Run the Gradle getAssetIndex task first, then this script; getAssets
will hash-check and reuse these objects without contacting the HTTP endpoint.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path,
                        default=Path.home() / ".gradle/caches/minecraft/assets")
    parser.add_argument("--index", default="1.11.json")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    index = args.cache / "indexes" / args.index
    objects = json.loads(index.read_text())["objects"]
    unique = {row["hash"]: row.get("size") for row in objects.values()}

    def fetch(item):
        digest, expected_size = item
        target = args.cache / "objects" / digest[:2] / digest
        if target.exists() and target.stat().st_size == expected_size:
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://resources.download.minecraft.net/{digest[:2]}/{digest}"
        fd, tmp_name = tempfile.mkstemp(prefix=digest + ".", dir=str(target.parent))
        try:
            h = hashlib.sha1()
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as src:
                while True:
                    block = src.read(1024 * 1024)
                    if not block:
                        break
                    h.update(block)
                    out.write(block)
            if h.hexdigest() != digest:
                raise RuntimeError(f"SHA-1 mismatch for {url}")
            os.replace(tmp_name, target)
            return True
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    downloaded = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for changed in pool.map(fetch, unique.items()):
            downloaded += int(changed)
    print(f"verified {len(unique)} Minecraft assets; downloaded {downloaded}")


if __name__ == "__main__":
    main()
