#!/usr/bin/env python3
"""
Apple Vision OCR prototype for RoboSlate-arm.

Runs VNRecognizeTextRequest on one image or a directory of images.
Prints each detected text blob (text, confidence, bbox) and saves
annotated copies to output/ for visual inspection.

Usage:
    python vision_test.py /path/to/frame.jpg
    python vision_test.py /path/to/frames/dir/
"""

import os
import sys

try:
    import Vision
    import Quartz
except ImportError:
    print("ERROR: pyobjc-framework-Vision not installed.")
    print("Run:  pip install pyobjc-framework-Vision pyobjc-framework-Quartz")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: Pillow not installed.")
    print("Run:  pip install Pillow")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Apple Vision OCR
# ---------------------------------------------------------------------------

def run_ocr(image_path):
    """
    Run VNRecognizeTextRequest on a single image file.

    Returns a list of dicts:
        {
            "text":       str,
            "confidence": float (0-1),
            "bbox":       (x, y, w, h)  — normalised, TOP-LEFT origin
        }

    Apple Vision uses CoreGraphics coordinates (bottom-left origin).
    We flip Y so the bbox matches our standard top-left convention:
        top_left_y = 1 - cg_y - height
    """
    image_url = Quartz.NSURL.fileURLWithPath_(image_path)

    request = Vision.VNRecognizeTextRequest.alloc().init()
    # Accurate mode uses the Neural Engine — slower but much better
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    # Language correction would mangle codes like "A051", "INT 42", "sc."
    request.setUsesLanguageCorrection_(False)

    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(image_url, {})
    success, error = handler.performRequests_error_([request], None)

    if not success:
        print(f"  Vision error: {error}")
        return []

    results = []
    for obs in (request.results() or []):
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        best = candidates[0]
        text = best.string()
        confidence = float(best.confidence())

        # CoreGraphics bbox: bottom-left origin, normalised
        cg = obs.boundingBox()
        x = float(cg.origin.x)
        cg_y = float(cg.origin.y)
        w = float(cg.size.width)
        h = float(cg.size.height)
        # Flip to top-left origin
        y = 1.0 - cg_y - h

        results.append({
            "text":       text,
            "confidence": confidence,
            "bbox":       (x, y, w, h),
        })

    # Sort top-to-bottom, left-to-right
    results.sort(key=lambda r: (round(r["bbox"][1] * 10), r["bbox"][0]))
    return results


# ---------------------------------------------------------------------------
# Annotated image output
# ---------------------------------------------------------------------------

def save_annotated(image_path, detections, output_dir):
    """
    Draw bounding boxes and text labels on a copy of the image.
    Saves to output_dir/<original_stem>_annotated.jpg.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Try to get a reasonable font; fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=max(14, h // 60))
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=max(11, h // 80))
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    for det in detections:
        bx, by, bw, bh = det["bbox"]
        conf = det["confidence"]
        text = det["text"]

        # Pixel coords
        x0 = int(bx * w)
        y0 = int(by * h)
        x1 = int((bx + bw) * w)
        y1 = int((by + bh) * h)

        # Colour by confidence: green=high, orange=medium, red=low
        if conf >= 0.9:
            colour = (50, 200, 50)
        elif conf >= 0.7:
            colour = (230, 150, 30)
        else:
            colour = (220, 50, 50)

        draw.rectangle([x0, y0, x1, y1], outline=colour, width=2)

        label = f"{text}  ({conf:.2f})"
        # Background pill behind label for readability
        bbox_text = draw.textbbox((x0, y0 - 2), label, font=small_font)
        draw.rectangle(bbox_text, fill=(0, 0, 0, 180))
        draw.text((x0, y0 - 2), label, fill=colour, font=small_font)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{stem}_annotated.jpg")
    img.save(out_path, "JPEG", quality=90)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_image(image_path, output_dir):
    print(f"\n{'='*60}")
    print(f"  {os.path.basename(image_path)}")
    print(f"{'='*60}")

    detections = run_ocr(image_path)

    if not detections:
        print("  (no text detected)")
        return

    print(f"  {'TEXT':<30}  {'CONF':>5}  BBOX (x, y, w, h)")
    print(f"  {'-'*30}  {'-'*5}  {'-'*30}")
    for det in detections:
        bx, by, bw, bh = det["bbox"]
        print(f"  {det['text']:<30}  {det['confidence']:>5.2f}  "
              f"({bx:.2f}, {by:.2f}, {bw:.2f}, {bh:.2f})")

    out_path = save_annotated(image_path, detections, output_dir)
    print(f"\n  Annotated: {out_path}")


def collect_images(path):
    exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.splitext(f)[1].lower() in exts
        )
    return []


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    images = []
    for arg in sys.argv[1:]:
        images.extend(collect_images(arg))

    if not images:
        print("No images found.")
        sys.exit(1)

    print(f"Processing {len(images)} image(s)...")
    for img_path in images:
        process_image(img_path, output_dir)

    print(f"\nAnnotated images saved to: {output_dir}/")


if __name__ == "__main__":
    main()
