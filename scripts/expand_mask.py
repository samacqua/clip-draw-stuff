"""Expand a mask by a given number of pixels using morphological dilation.

Usage:
    python scripts/expand_mask.py imgs/masks/prof-mask-manual.png 100 --output imgs/masks/prof-mask-manual4.png
    python scripts/expand_mask.py imgs/masks/prof-mask-manual.png 50 --blur 45 --clear-top 200 --clear-right 200
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter
from scipy.ndimage import binary_dilation


def main():
    parser = argparse.ArgumentParser(description="Expand a mask via morphological dilation")
    parser.add_argument("input", help="Input mask image (grayscale or alpha channel)")
    parser.add_argument("radius", type=int, help="Dilation radius in pixels")
    parser.add_argument("--output", "-o", help="Output path (default: <input>-expanded-<radius>.png)")
    parser.add_argument("--blur", type=int, default=45, help="Gaussian blur radius for soft edges (default: 45)")
    parser.add_argument("--clear-top", type=int, default=0, help="Zero out this many rows from the top")
    parser.add_argument("--clear-right", type=int, default=0, help="Zero out this many cols from the right (upper half only)")
    parser.add_argument("--preview", help="Photo to generate an applied preview with")
    parser.add_argument("--inverse", action="store_true", help="Also save the inverted mask")
    args = parser.parse_args()

    mask = Image.open(args.input).convert("L")
    arr = np.array(mask)

    binary = arr > 128
    diameter = 2 * args.radius + 1
    struct = np.ones((diameter, diameter))
    dilated = binary_dilation(binary, structure=struct, iterations=1)
    arr_out = (dilated.astype(np.float32) * 255).astype(np.uint8)

    result = Image.fromarray(arr_out)
    if args.blur > 0:
        result = result.filter(ImageFilter.GaussianBlur(args.blur))

    arr_result = np.array(result)
    rows, cols = arr_result.shape
    if args.clear_top > 0:
        arr_result[: args.clear_top, :] = 0
    if args.clear_right > 0:
        arr_result[: rows // 2, cols - args.clear_right :] = 0
    result = Image.fromarray(arr_result)

    inp = Path(args.input)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = inp.parent / f"{inp.stem}-expanded-{args.radius}{inp.suffix}"

    result.save(out_path)
    print(f"Saved mask: {out_path}")

    if args.inverse:
        inv_path = out_path.parent / out_path.name.replace("mask", "nonmask")
        ImageChops.invert(result).save(inv_path)
        print(f"Saved inverse: {inv_path}")

    if args.preview:
        photo = Image.open(args.preview).convert("RGBA")
        rgba = photo.copy()
        rgba.putalpha(result.resize(photo.size, Image.LANCZOS))
        preview_path = out_path.with_name(out_path.stem + "-applied" + out_path.suffix)
        rgba.save(preview_path)
        print(f"Saved preview: {preview_path}")


if __name__ == "__main__":
    main()
