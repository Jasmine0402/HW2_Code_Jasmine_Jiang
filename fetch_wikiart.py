"""Extract complete style directories from the specified Kaggle ZIP by byte range."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
from pathlib import Path
import subprocess
import zipfile

from PIL import Image, ImageOps

URL = "https://www.kaggle.com/api/v1/datasets/download/steubk/wikiart?datasetVersionNumber=1"
SIZE = 33_662_671_673
TAIL_SIZE = 25_000_000
STYLES = ("Abstract_Expressionism", "Color_Field_Painting", "Minimalism")


def download(path, start, stop):
    expected = stop - start
    if path.is_file() and path.stat().st_size == expected:
        return
    temporary = path.with_suffix(".part")
    existing = temporary.stat().st_size if temporary.exists() else 0
    if existing > expected:
        raise IOError(f"Oversized partial download: {temporary}")
    if existing < expected:
        suffix = path.with_suffix(".suffix")
        length = expected - existing
        subprocess.run(
            ["curl", "--location", "--fail", "--silent", "--show-error", "--retry", "3",
             "--speed-limit", "4096", "--speed-time", "40", "--max-time", "300",
             "--max-filesize", str(length), "--range", f"{start + existing}-{stop - 1}",
             URL, "--output", str(suffix)], check=True
        )
        if suffix.stat().st_size != length:
            raise IOError(f"Unexpected range length for {path}")
        with temporary.open("ab") as target, suffix.open("rb") as stream:
            shutil.copyfileobj(stream, target)
        suffix.unlink()
    temporary.replace(path)
    print(f"Downloaded {path.name} ({expected:,} bytes)", flush=True)



class RangeFile:
    def __init__(self, regions):
        self.regions = regions
        self.offset = 0

    def seek(self, offset, whence=os.SEEK_SET):
        self.offset = (0, self.offset, SIZE)[whence] + offset
        return self.offset

    def tell(self):
        return self.offset

    def seekable(self):
        return True

    def read(self, size=-1):
        length = SIZE - self.offset if size < 0 else size
        if length == 0:
            return b""
        for start, stop, path in self.regions:
            if start <= self.offset and self.offset + length <= stop:
                with path.open("rb") as stream:
                    stream.seek(self.offset - start)
                    data = stream.read(length)
                self.offset += len(data)
                return data
        raise IOError(f"Missing ZIP range at {self.offset:,} for {length:,} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    tail_start = SIZE - TAIL_SIZE
    tail = args.cache / "zip_tail.bin"
    download(tail, tail_start, SIZE)
    regions = [(tail_start, SIZE, tail)]
    with zipfile.ZipFile(RangeFile(regions)) as archive:
        members = sorted(archive.infolist(), key=lambda member: member.header_offset)
    image_ext = {".jpeg", ".jpg", ".png"}
    selected = []
    for style in STYLES:
        inds = [
            i for i, entry in enumerate(members)
            if entry.filename.startswith(style + "/")
            and Path(entry.filename).suffix.lower() in image_ext
        ]
        if not inds:
            raise ValueError(f"No ZIP entries for {style}")
        selected.extend(members[i] for i in inds)
        start = members[inds[0]].header_offset
        stop = members[inds[-1] + 1].header_offset
        regions.append((start, stop, args.cache / f"{style}.bin"))
    chunk_size = 48 * 1024 * 1024
    files_to_join = []
    jobs = []
    for start, stop, path in regions[1:]:
        if path.is_file() and path.stat().st_size == stop - start:
            continue
        chunks = []
        for offset in range(start, stop, chunk_size):
            chunk = path.with_suffix(f".{offset}.chunk")
            jobs.append((chunk, offset, min(offset + chunk_size, stop)))
            chunks.append(chunk)
        files_to_join.append((path, chunks))
    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(lambda job: download(*job), jobs))
    for path, chunks in files_to_join:
        temporary = path.with_suffix(".join")
        with temporary.open("wb") as target:
            for chunk in chunks:
                with chunk.open("rb") as stream:
                    shutil.copyfileobj(stream, target)
        temporary.replace(path)
    with zipfile.ZipFile(RangeFile(regions)) as archive:
        for i, entry in enumerate(selected, 1):
            destination = args.out / entry.filename
            destination = destination.with_suffix(destination.suffix + ".png")
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as raw, Image.open(raw) as image:
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    rgb.resize((64, 64), Image.Resampling.LANCZOS).save(destination)
            if i % 500 == 0 or i == len(selected):
                print(f"Prepared {i:,}/{len(selected):,} images", flush=True)


if __name__ == "__main__":
    main()
