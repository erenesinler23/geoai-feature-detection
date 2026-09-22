from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_control_test(image_path: str, detector, confidence_threshold: float,
                      max_plausible_fraction: float, out_path: str) -> dict:
    """Inspect model boxes on a photograph without claiming accuracy."""
    image = Image.open(image_path).convert("RGB")
    raw = detector.detect_pixels(image)

    keep = raw["scores"] >= confidence_threshold
    boxes = raw["boxes"][keep]
    scores = raw["scores"][keep]
    labels = [label for label, k in zip(raw["labels"], keep.tolist()) if k]

    img_w, img_h = image.size
    plausible_count = 0
    rows = []

    for pt_box, score, label in zip(boxes, scores, labels):
        x_min, y_min, x_max, y_max = pt_box.tolist()
        width_frac = (x_max - x_min) / img_w
        height_frac = (y_max - y_min) / img_h
        is_plausible = width_frac < max_plausible_fraction and height_frac < max_plausible_fraction
        plausible_count += is_plausible
        rows.append({
            "box": pt_box.tolist(),
            "score": float(score), "label": detector.text_queries[label],
            "width_frac": width_frac, "height_frac": height_frac, "plausible": is_plausible,
        })

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    for pt_box, row in zip(boxes, rows):
        x_min, y_min, x_max, y_max = pt_box.tolist()
        color = "lime" if row["plausible"] else "red"
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    fill=False, edgecolor=color, linewidth=1, alpha=0.7))
    ax.set_title(f"Control test, {plausible_count}/{len(rows)} plausible detections "
                 f"(green = plausible, red = oversized)")
    ax.axis("off")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()

    from src.archive import write_json
    write_json(str(out_path) + ".json", {"raw": detector.last_raw,
               "confidence_threshold": confidence_threshold, "text_queries": detector.text_queries,
               "model_revision": detector.model_revision, "max_plausible_fraction": max_plausible_fraction,
               "detections": rows, "interpretation": "Size check only, no accuracy labels"})
    return {"total_detections": len(rows), "plausible_count": plausible_count, "detections": rows}