# Revision notes

## Documentation update on 22 September 2026

The README now includes a link to the paper, explains the difference between the benchmark and Nottingham shortcut settings, and separates setup instructions from research findings. The test instructions include both the app tests and the separate hybrid checks. Five older docstrings now use short descriptions consistent with the rest of the code.

This update changes documentation and docstrings only. Detection, matching and saved results are unchanged. The existing verification log is retained as the record of the earlier checks. A new .gitignore excludes Python caches, macOS metadata, virtual environments and generated outputs from future Git tracking.

## Corrected evaluation

The corrected experiment covers seven locations with archived imagery, OSM footprints, predictions and settings. These runs supply the paper tables. Historical figures retain the earlier outputs and their original labels.

Evaluation uses the usable image area, unique matches based on IoU, size checks in metres and explicit values for empty results. The imagery reader checks crop coverage and scene classification. The OSM parser includes multipolygon buildings, and the pipeline repairs invalid geometry. The app saves each completed run separately and supports repeated place searches.

At IoU 0.25, the colour rule achieved precision between 4.84 and 24.94 percent. London was the only benchmark crop with a retained model box and hybrid candidates; none matched an eligible building. Several parts of the experiment changed from the earlier version, so the difference from the old intersection scores cannot be attributed to matching alone.

## Recorded verification and later diagnostics

The earlier verification completed 18 automated unittest checks and two separate hybrid checks. All seven archived evaluations reproduced their retained and matched counts. The synthetic demo completed, and the Prague photograph check returned 124 candidates with 93 passing its relative size rule.

The diagnostics on 21 September added a coordinate check, a bird photograph control, a generic building prompt comparison and a degradation test on three external overhead images. The model located both birds and matched buildings in two external images. Degrading the overhead images removed all building matches. These controls are stored in a separate diagnostic archive.

The study remains exploratory. It needs independently checked roof colour labels and fresh evaluation scenes. The small degradation test investigates image detail, while a realistic study of sensor resolution remains future work.

## Data and attribution

Building data comes from OpenStreetMap contributors. Imagery is Copernicus Sentinel data accessed through Microsoft Planetary Computer. The model is Google's OWL ViT checkpoint. The paper retains its references, code examples and supplied figures.

Keep the original experiment archives and verification log intact. They record the code and data used at the time of each run, including versions that differ from the current source.
