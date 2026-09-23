#!/usr/bin/env python3
"""Create a COCO annotation file limited to the image names in a text file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--annotations", type=Path, required=True)
parser.add_argument("--images-file", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)


def main() -> None:
    args = parser.parse_args()
    source = json.loads(args.annotations.read_text(encoding="utf-8"))
    wanted = {line.strip() for line in args.images_file.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.lstrip().startswith("#")}
    # COCO often stores paths such as ``test2017/name.jpg`` while the
    # comparison list intentionally stores portable bare filenames.
    images = [image for image in source["images"]
              if image["file_name"] in wanted or Path(image["file_name"]).name in wanted]
    found = {Path(image["file_name"]).name for image in images}
    missing = wanted - found
    if missing:
        raise ValueError(f"Images absent from COCO annotations: {sorted(missing)}")
    image_ids = {image["id"] for image in images}
    subset = dict(source)
    subset["images"] = images
    subset["annotations"] = [ann for ann in source["annotations"] if ann["image_id"] in image_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(images)} images and {len(subset['annotations'])} annotations to {args.output}")


if __name__ == "__main__":
    main()
