# FC-Wiring

Vision inspection for FC connector wiring. It detects the wire colors in each inspection region and checks that they sit in the configured left-to-right order. For example, profile `FCFK2FF3` expects the black wire on the left and the blue wire on the right at both connectors.

## Setup

```
pip install -r requirements.txt
python main.py
```

## Using the app

1. **Open Image** or **Start Camera** (set the camera index with `Cam #`).
2. Choose a **Profile** (part model). The verdict panel shows PASS/FAIL, and the table shows the expected and detected order for each region.
3. With **Live inspect** on, every camera frame is inspected automatically. **Save Result** writes the annotated image.

### Configuring a new model

1. Load a reference image, click **New**, and enter the model name.
2. Turn on **Edit regions** and drag a box around the wire exit of each connector. Keep the box tight: it should cover the wires only, not the blue fixture.
3. Enter the expected colors left to right, e.g. `black, blue`.
4. Click **Save Config**.

## How detection works

Each region is converted to HSV and masked for each expected color (ranges are in `config.json`). Wires run vertically through the region, so the detector looks for narrow columns where the mask covers most of the region's height. It then takes the peak of those columns as the wire's position. Partial blobs such as the black rubber boot or the foam sleeve get rejected this way. The wire centers are sorted by x and compared with `expected_order`.

The settings you can tune in `config.json`:

- `colors.<name>.hsv_ranges`: OpenCV HSV bounds (H 0–180). Add new wire colors here.
- `detection.min_column_fill`: fraction of the region height a column must cover to count as wire.
- `profiles.<model>.image_size`: the resolution the boxes were drawn at. Boxes scale automatically to other resolutions.

## Tests

```
python -m pytest
```
