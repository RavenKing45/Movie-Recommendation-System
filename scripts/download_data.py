"""Download MovieLens 32M into data/raw/ and verify its checksum."""
import hashlib
import re
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
RAW = Path("data/raw")
ZIP = RAW / "ml-32m.zip"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    if not ZIP.exists():
        print("Downloading ml-32m.zip (~239 MB)...")
        urllib.request.urlretrieve(URL, ZIP)

    actual = md5(ZIP)
    try:
        text = urllib.request.urlopen(URL + ".md5").read().decode()
        expected = re.search(r"[0-9a-fA-F]{32}", text).group(0).lower()
        status = "OK" if actual == expected else "MISMATCH"
        print(f"MD5 {status}: {actual}")
        if actual != expected:
            raise SystemExit("Checksum mismatch, delete the zip and rerun.")
    except Exception as e:
        if isinstance(e, SystemExit):
            raise
        print(f"Could not fetch published checksum ({e}); computed MD5: {actual}")

    print("Extracting...")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(RAW)
    print("Done:", sorted(p.name for p in (RAW / "ml-32m").iterdir()))


if __name__ == "__main__":
    main()