# FC-Wiring

Vision inspection for FC connector wiring. It detects the wire colors in each inspection region and checks that they sit in the configured left-to-right order. For example, profile `FCFK2FF3` expects the black wire on the left and the blue wire on the right at both connectors.

## Setup

```
pip install -r requirements.txt
python main.py
```

## Using the app

1. **Open Images** (select one or more files), **Open Folder**, or use the camera bar: pick a detected camera from the **Camera** list, choose a **Resolution**, and press **Start Camera**. Press **Refresh** after plugging in a camera. Changing the camera or resolution while it is running restarts it.
2. Choose a **Profile** (part model). The verdict panel shows PASS/FAIL, and the table shows the expected and detected order for each region.
3. With **Live inspect** on, every camera frame is inspected automatically. **Save Result** writes the annotated image.

### Testing with images

After loading several images or a folder, step through them with **< Prev** / **Next >** (or the arrow keys). Each image is inspected as it is shown.

**Test All Images** runs every loaded image against the current profile and opens a summary window with the pass count and the result for each image. Double-click a row to view that image, or use **Export CSV** to save the report.

### Configuring a new model

1. Load a reference image, click **New**, and enter the model name.
2. Turn on **Edit regions** and drag a box around the wire exit of each connector. Keep the box tight: it should cover the wires only, not the blue fixture.
3. Enter the expected colors left to right, e.g. `black, blue` or, for a 3 or 4 wire connector, `black, blue, black, red`. A color can appear more than once.
4. Click **Save Config**.

## How detection works

Each region is converted to HSV and masked for each expected color (ranges are in `config.json`). Wires run vertically through the region, so the detector looks for narrow columns where the mask covers most of the region's height. Each run of such columns is split around its fill peaks, and for a color expected *n* times the *n* pieces with the highest fill are kept as wires. Partial blobs such as the black rubber boot or the foam sleeve get rejected this way. Two neighbouring wires of the same color need a visible gap or shadow between them to be counted separately. The wire centers are sorted by x and compared with `expected_order`.

The settings you can tune in `config.json`:

- `colors.<name>.hsv_ranges`: OpenCV HSV bounds (H 0–180). Add new wire colors here.
- `detection.min_column_fill`: fraction of the region height a column must cover to count as wire.
- `profiles.<model>.image_size`: the resolution the boxes were drawn at. Boxes scale automatically to other resolutions.

## Tests

```
python -m pytest
```
