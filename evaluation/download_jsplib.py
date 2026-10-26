"""Download and verify files from a pinned, previously retrieved GitHub index."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    root = args.snapshot.resolve()
    commit = root.name
    index = json.loads((root / "upstream-json-index.json").read_text(encoding="utf-8"))
    output = root / "raw"
    output.mkdir(exist_ok=True)
    def fetch(entry):
        name = entry["name"]
        if Path(name).name != name or not re.fullmatch(r"[A-Za-z0-9_-]+\.json", name):
            raise ValueError("Unsafe source filename")
        path = output / name
        url = f"https://raw.githubusercontent.com/ScheduleOpt/benchmarks/{commit}/jobshop/instances/json/{name}"
        def valid(data):
            return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == entry["sha"]
        if path.exists() and valid(path.read_bytes()):
            return {"file": name, "state": "verified_cached", "git_sha": entry["sha"]}
        error = None
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers={"User-Agent": "CEJSP-data-import"}), timeout=40) as stream:
                    data = stream.read()
                if not valid(data):
                    raise ValueError("Downloaded content does not match pinned Git blob SHA")
                temporary = path.with_suffix(".json.part")
                temporary.write_bytes(data)
                temporary.replace(path)
                return {"file": name, "state": "downloaded_verified", "git_sha": entry["sha"], "sha256": hashlib.sha256(data).hexdigest()}
            except Exception as exc:
                error = str(exc)
                if attempt < 2:
                    time.sleep(1)
        return {"file": name, "state": "failed", "error": error}
    entries = [e for e in index if e["name"].endswith(".json")]
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(fetch, e) for e in entries]
        for future in as_completed(jobs):
            results.append(future.result())
            if len(results) % 30 == 0:
                print(f"Verified downloads: {len(results)}/{len(entries)}", flush=True)
    report = {"commit": commit, "downloaded_or_cached": sum(r["state"] != "failed" for r in results),
              "failed": sum(r["state"] == "failed" for r in results),
              "archives_not_downloaded": [e["name"] for e in index if not e["name"].endswith(".json")],
              "files": sorted(results, key=lambda r: r["file"])}
    (root / "download-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("commit", "downloaded_or_cached", "failed")}), flush=True)
    raise SystemExit(bool(report["failed"]))


if __name__ == "__main__":
    main()
