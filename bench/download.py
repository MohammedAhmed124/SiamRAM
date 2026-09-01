"""Fetch single-object-tracking benchmarks onto a headless machine.

Usage: python bench/download.py --dataset uav123 --dest /vol/bench
See bench/DATASETS.md for sources, layouts and the manual steps.
"""

import argparse
import os
import re
import sys
import urllib.request
import zipfile

# Exact byte sizes read from the hosts' Content-Range headers.
DATASETS = {
    "uav123": {
        "url": "https://drive.google.com/file/d/0B6sQMCU1i4NbNGxWQzRVak5yLWs/view?usp=drivesdk&resourcekey=0-IjwQcWEzP2x3ec8kXtLBpA",
        "file": "Dataset_UAV123.zip",
        "size": 14049397769,
        "root": "UAV123",
    },
    "uav123_10fps": {
        "url": "https://drive.google.com/file/d/0B6sQMCU1i4NbZmFlQmJBVDlLRDg/view?usp=drivesdk&resourcekey=0--jsSKS1oGFidNhgMF75cSQ",
        "file": "Dataset_UAV123_10fps.zip",
        "size": 4688528930,
        "root": "UAV123_10fps",
    },
    "visdrone_sot": {
        "url": "https://drive.google.com/file/d/1xCiHjU4JlR9QsYtiHYy2UUd3m6NthoBC/view",
        "file": "VisDrone2019-SOT-test-dev.zip",
        "size": 12097003333,
        "root": "VisDrone2019-SOT-test-dev",
    },
}


LASOT_SIZE_WARNING = """LaSOT publishes no testing-set-only archive: the 70 per-category zips each
mix the 16 training and 4 testing sequences of that category, so all 248 GB come
down for the 280 sequences we evaluate. Delete the training sequences afterwards
if space is tight - bench/datasets.py reads testing_set.txt and ignores them."""

MANUAL = {
    "dtb70": """DTB70 images are only on Baidu Pan (no mirror found):
  https://pan.baidu.com/s/1SftGHD7SyIFyBXExHbbYAQ   (via https://github.com/flyers/drone-tracking)
Download DTB70.zip on a machine with a Baidu account, then upload it to the
volume and unzip to <dest>/DTB70. The 70 ground-truth and attribute files are
public and can be fetched without Baidu:
  git clone --depth 1 https://github.com/flyers/drone-tracking
  # experiments/anno/<Seq>.txt  == <Seq>/groundtruth_rect.txt
  # experiments/anno/att/<Seq>.txt == 11 per-sequence attribute flags""",
}


def drive_uc_url(url):
    """Any Google Drive link normalised to uc?id=..., keeping the resourcekey."""
    ids = re.search(r"/d/([A-Za-z0-9_-]+)|[?&]id=([A-Za-z0-9_-]+)", url)
    if not ids:
        raise SystemExit(f"no Drive file id in {url}")
    fid = next(g for g in ids.groups() if g)
    key = re.search(r"[?&]resourcekey=([A-Za-z0-9_-]+)", url)
    return f"https://drive.google.com/uc?id={fid}" + (f"&resourcekey={key.group(1)}" if key else "")


def _drive_get(url):
    """Streaming response for a Drive file, past the large-file virus-scan interstitial.

    gdown 6 rebuilds every link as uc?id=<id> and drops the resourcekey, which the
    pre-2021 UAV123 links need - without it Drive serves a sign-in page.
    """
    import requests

    session = requests.Session()
    r = session.get(drive_uc_url(url), stream=True, timeout=60)
    if not r.headers.get("Content-Type", "").startswith("text/html"):
        return r
    html = r.text
    if "Quota exceeded" in html:
        raise SystemExit(f"Google Drive download quota exceeded for {url}; retry in a few hours")
    action = re.search(r'<form[^>]+action="([^"]+)"', html)
    if not action:
        raise SystemExit(f"no confirm form at {url}; the link may be private or removed")
    fields = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', html))
    r = session.get(action.group(1).replace("&amp;", "&"), params=fields, stream=True, timeout=60)
    if r.headers.get("Content-Type", "").startswith("text/html"):
        why = "quota exceeded" if "Quota exceeded" in r.text else "an unexpected page"
        raise SystemExit(f"Google Drive returned {why} for {url}")
    return r


def _download(url, path, size=None):
    """Fetch url to path, handling Google Drive's confirm step."""
    if size and os.path.exists(path) and os.path.getsize(path) == size:
        print(f"have {path}")
        return
    print(f"downloading {url} -> {path}")
    if "drive.google.com" in url:
        r = _drive_get(url)
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(path, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total and done % (1 << 28) < (1 << 20):
                    print(f"  {done / total:.0%} ({done >> 20} MiB)", flush=True)
    else:
        urllib.request.urlretrieve(url, path)
    got = os.path.getsize(path)
    if size and got != size:
        raise SystemExit(f"size mismatch for {path}: expected {size}, got {got}")


def _extract(path, dest):
    print(f"extracting {path}")
    with zipfile.ZipFile(path) as z:
        z.extractall(dest)


def _lasot_seq_name(member, cat):
    """The <cat>-<n> sequence a zip member belongs to, whether or not the zip nests under <cat>/."""
    for part in member.split("/"):
        if part.startswith(cat + "-") and part[len(cat) + 1:].isdigit():
            return part
    return None


def _fetch_lasot(dest):
    """Pull the 70 per-category LaSOT zips off the authors' Hugging Face mirror, keeping test only.

    Each category zip holds 16 training and 4 testing sequences. Only the testing_set.txt names
    are extracted and each zip is deleted straight after, so the volume holds ~50 GB not ~248 GB.
    """
    from huggingface_hub import hf_hub_download

    print(LASOT_SIZE_WARNING, file=sys.stderr)
    root = os.path.join(dest, "LaSOT")
    os.makedirs(root, exist_ok=True)
    listing = hf_hub_download("l-lt/LaSOT", "testing_set.txt", repo_type="dataset", local_dir=root)
    keep = {n.strip() for n in open(listing, encoding="utf-8-sig") if n.strip()}
    cats = sorted({n.rsplit("-", 1)[0] for n in keep})
    print(f"{len(keep)} test sequences across {len(cats)} categories")
    for i, cat in enumerate(cats, 1):
        wanted = {n for n in keep if n.rsplit("-", 1)[0] == cat}
        if all(os.path.isdir(os.path.join(root, cat, n)) for n in wanted):
            print(f"[{i}/{len(cats)}] have {cat}")
            continue
        print(f"[{i}/{len(cats)}] {cat}")
        archive = hf_hub_download("l-lt/LaSOT", f"{cat}.zip", repo_type="dataset", local_dir=root)
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.namelist() if _lasot_seq_name(m, cat) in wanted]
            if not members:
                raise RuntimeError(f"{cat}.zip contained none of {sorted(wanted)}")
            # Zips that already nest under <cat>/ extract to root; flat ones get it added.
            nested = members[0].startswith(cat + "/")
            z.extractall(root if nested else os.path.join(root, cat), members)
        os.remove(archive)


def _fetch_trackingnet(dest):
    """Pull the 511-sequence TrackingNet TEST split off the authors' Hugging Face mirror."""
    from huggingface_hub import snapshot_download

    root = os.path.join(dest, "TrackingNet")
    snapshot_download("SilvioGiancola/TrackingNet", repo_type="dataset", local_dir=root,
                      allow_patterns="TEST/*")
    zips = os.path.join(root, "TEST", "zips")
    for name in sorted(os.listdir(zips)):
        out = os.path.join(root, "TEST", "frames", name[: -len(".zip")])
        if not os.path.isdir(out):
            _extract(os.path.join(zips, name), out)


CUSTOM = {
    "lasot": _fetch_lasot,
    "trackingnet": _fetch_trackingnet,
}


def fetch(name, dest):
    os.makedirs(dest, exist_ok=True)
    if name in MANUAL:
        print(MANUAL[name], file=sys.stderr)
        return 1
    if name in CUSTOM:
        CUSTOM[name](dest)
        return 0
    spec = DATASETS[name]
    archive = os.path.join(dest, spec["file"])
    _download(spec["url"], archive, spec["size"])
    if not os.path.isdir(os.path.join(dest, spec["root"])):
        _extract(archive, dest)
    return 0


def main():
    names = sorted(set(DATASETS) | set(MANUAL) | set(CUSTOM))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=names)
    p.add_argument("--dest", required=True)
    a = p.parse_args()
    return fetch(a.dataset, a.dest)


if __name__ == "__main__":
    sys.exit(main())
