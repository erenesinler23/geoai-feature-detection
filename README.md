# GeoAI building comparison

[Read the research paper](docs/lutfieren-esinler-geoai-research-paper.pdf?raw=true)

This project grew out of an attempt to find red roofs in satellite images. It compares a colour rule, OWL ViT and a simple hybrid using Sentinel 2 imagery. The Flask app lets you search for a place, inspect predictions and compare them with OpenStreetMap building footprints.

The paper is titled *Comparing Colour Thresholding and a Vision Language Model for Building Matching in Satellite Images*. The evaluation checks how well predictions overlap mapped buildings. Roof colour remains unverified because the reference data includes buildings of every colour.

## Setup

The recorded environment used Python 3.14.7 and the versions in requirements.txt. Create a local environment, install the packages and start the app:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5055 in your browser. The first analysis downloads the model unless it is already cached. Live searches need internet access for imagery, place lookup and building data.

In VS Code, select the interpreter inside `.venv` so the editor uses the same packages as the terminal. OWL ViT loads once per application process and selects Apple MPS, then CUDA, then CPU according to availability. The terminal prints the device it actually uses.

## Using the app

Enter a place name or choose a Quick Access location. Searches work beyond the seven cities used in the paper. Each search analyses a crop around the returned coordinates, so a city name selects a local scene rather than its whole administrative area.

Repeated searches reuse results held in memory. Choose Rerun live to fetch and analyse again. Restarting the app clears this memory, while completed run folders stay on disk. Live analyses run one at a time; cached pages and saved images can still be served to other requests.

## How the comparison works

The default crop is 300 by 300 pixels. The imagery reader checks that the crop fits inside the tile and uses scene classification to mask clouds, shadows, snow and missing pixels. It retains classification codes 4, 5, 6 and 7 and accepts at most 10 percent masked pixels. Classification errors can still affect the usable area.

The pipeline requests OSM building ways and multipolygon relations within the crop bounds. It evaluates complete footprints inside the usable area and records how many buildings were excluded at the boundary or mask. Predictions are clipped to the same usable area. A prediction of an excluded partial building may consequently remain unmatched.

Each prediction can match one reference building, and each reference can match one prediction. Matching first maximises the number of pairs meeting the IoU threshold, then their total overlap. The default IoU is 0.25. Replay also evaluates 0.10 and 0.50. Duplicate predictions remain in the evaluation and cannot earn repeated credit for the same building.

Before matching, the evaluator removes regions wider or taller than 500 metres and records their count. This filters very large regions; smaller boxes can still contain several buildings or background features. All reported scores apply after this filter.

Precision is the matched share of retained predictions. Recall is the matched share of eligible reference buildings. False discovery rate is the unmatched share of retained predictions. F1 is twice the match count divided by the sum of predictions and references. Scores are percentages. An empty denominator appears as null in JSON and N/A in the app. When references exist but predictions are empty, recall and F1 are zero.

The colour rule uses the visual RGB asset. These display values are different from calibrated reflectance. OSM roof colour tags are retained when present, but the study has no independently checked colour labels. Map omissions, inaccurate footprints and differences between map and image dates also affect the comparison.

## Running experiments

Run one analysis using config.py:

```sh
python main.py
```

Run the fixed batch, or select locations from it:

```sh
python run_benchmark.py
python run_benchmark.py --locations london prague
```

The benchmark uses the shared colour thresholds 120, 25 and 25 and model confidence 0.01. All seven cities were explored before this batch, so it is an exploratory comparison. Nottingham's Quick Access shortcut retains its earlier settings of 140, 40 and 40 and confidence 0.02. Use benchmark.json to reproduce the paper's setup.

Each live run gets a new folder containing the image, mask, reference footprints, predictions, plots, raw model output, settings, dependency versions, source snapshot and checksums. The manifest records the STAC scene and OSM snapshot timestamp. The current archive writer removes signed download query parameters.

## Saved evidence and replay

The outputs folder is excluded from Git tracking to keep routine searches out of the source repository. Preserve the original run archives separately. Reproducing the paper requires the seven outputs/verified folders, while the London, Prague and Dubrovnik examples have their own saved run folders. The separate diagnostic archive contains the later controls and their scripts. Download links should accompany the published research release.

With the saved folders available, repeat matching or detection using:

```sh
python replay_evaluation.py outputs/verified/marrakech
python rerun_saved.py outputs/verified/marrakech outputs/replayed/marrakech
```

Replay writes sensitivity.json into the supplied folder, so use a working copy to preserve the original evidence and checksums. The detection rerun currently loads the model revision selected by the detector's default or OWL_MODEL_REVISION. Check it against the saved manifest when reproducing a run.

The two summary files, results_summary.csv and threshold_sensitivity.csv, contain the values used for the paper tables. Historical images in legacy_figures retain the earlier settings and intersection based evaluation. Keep those images separate from the corrected benchmark results.

## Checks and diagnostics

Run the 18 unittest checks and the two separate hybrid checks:

```sh
python -m unittest discover -s tests -v
python tests/test_hybrid_logic.py
```

Run the synthetic example or the original Prague photograph check:

```sh
python demo.py
python run_control_test.py
```

The Prague photograph check returned 124 candidates, of which 93 passed the relative size rule. Those counts describe model output; the photograph has no manual labels for measuring detection accuracy.

The additional diagnostics on 21 September 2026 used CPU execution and the recorded model revision. A known pixel box converted to the expected projected coordinates within 0.001 metres. The model also placed a box around each bird in a control photograph, with both scores above 0.1.

On the recent London and Prague satellite crops, the original prompts each produced one candidate above 0.01 confidence. Both boxes covered almost the whole scene and failed the 500 metre size limit. Dubrovnik produced no candidate above the confidence threshold.

Changing the prompt to buildings generally produced one match from three retained predictions in London, two from four in Prague and none from one in Dubrovnik. Recall was 0.0113 percent, 0.0349 percent and zero respectively. Most reference buildings still had no match. This prompt changes the target from coloured roofs to buildings generally and was used only as a diagnostic.

All three images in an external dataset's small validation archive were tested at confidence 0.01 and box IoU 0.50. The model matched 0 of 5, 8 of 9 and 4 of 6 annotated buildings. Reducing each image from 500 by 500 pixels to 25 by 25, then enlarging it again, removed every match. Even in the original images, prediction confidence stayed below 0.05.

These checks point to image detail as one contributor to the results. Their small sample and unverified original ground sampling distances limit the conclusion. The resizing procedure also differs from the way a satellite sensor forms an image. Further evaluation needs fresh scenes and reference labels matched to the chosen detection target.

## Project context

I proposed and developed this project during independent study time in my internship at BİTES, an ASELSAN subsidiary. The NCIA document *GeoAI Solutions to Support Selected NATO Use Cases* provided wider context for geospatial feature extraction. The project was my own study, with no claim of commissioning, review or endorsement by either organisation.

## Sources

[External building dataset](https://huggingface.co/datasets/keremberke/satellite-building-segmentation), using data/valid-mini.zip.

[Bird control photograph](https://huggingface.co/datasets/Narsil/image_dummy/blob/main/parrots.png).

[Polish ministry notice dated 14 May 2026](https://www.gov.pl/web/rozwoj-technologia/komunikat-nr-71--2026), identifying NCIA market survey RFI-187262-GEOAI. The notice was checked on 21 September 2026.

The paper cites the [original NCIA request for information](docs/RFI-187262-GEOAI.pdf), dated 12 May 2026, document NCIA/ACQ/2026/07098. Building data comes from OpenStreetMap contributors. Copernicus Sentinel imagery is accessed through Microsoft Planetary Computer, and the model is Google's OWL ViT checkpoint.
