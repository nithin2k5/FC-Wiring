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
3. With **Live inspect** on, every camera frame is inspected automatically. **Save Result** writes the annotated image. To inspect a single shot instead, press **Capture**: the camera stops, the current frame is saved to the `captures/` folder and inspected, and it's added to the image list. Press **Start Camera** to go live again.
4. If the camera image is upside down or mirrored, tick **Flip up/down** and/or **Flip left/right**. The flip applies to camera frames and loaded images, and **Save Config** remembers it. Flipping left/right also reverses the wire order the app sees, so only use it to undo a mirrored camera. Set the flip before configuring regions: regions drawn on unflipped images won't find their connectors once the image is flipped.

### Testing with images

After loading several images or a folder, step through them with **< Prev** / **Next >** (or the arrow keys). Each image is inspected as it is shown.

**Test All Images** runs every loaded image against the current profile and opens a summary window with the pass count and the result for each image. Double-click a row to view that image, or use **Export CSV** to save the report.

### Configuring a new model

1. Load a reference image, click **New**, and enter the model name.
2. Turn on **Edit regions** and drag a box around the wire exit of each connector. Keep the box tight: it should cover the wires only, not the blue fixture.
3. Enter the expected colors left to right, e.g. `black, blue` or, for a 3 or 4 wire connector, `black, blue, black, red`. A color can appear more than once.
4. Next, the app asks for the connector: drag a tight box around the connector body (the green housing and black boot), not the wires or the fixture. This picture (the *anchor*) is how the region finds its connector anywhere in the frame. Press Esc to skip, and the region stays at a fixed position. To change it later, delete the region and add it again. Use a clean camera frame, not an annotated image.
5. Then it asks for the **search area**: drag a box that holds this connector and its wires wherever they can be, and no other connector. The connector is then only looked for inside its area, which also fixes which connector is connector 1 and which is connector 2. Press Esc to skip, and the connector is looked for anywhere in the frame. **Set area** redoes this for the selected region.
6. After each area, the app offers to start detection. Choose **No** to set up the next connector, then **Yes** (or untick **Edit regions**) to start. Click **Save Config** to keep the setup.

Made a mistake? **Undo** (or Ctrl+Z) steps back through region changes one at a time: a wrong area, connector or wires box, a deleted region, or a changed order or name. Each region has three boxes: the yellow wires box and the pink connector box are used for detection, and the light-blue area only limits where they're searched for. Draw all three on frames from your own camera; boxes copied from another camera setup will be in the wrong place.

## How detection works

Each region is converted to HSV and masked for each expected color (ranges are in `config.json`). Wires run vertically through the region, so the detector looks for narrow columns where the mask covers most of the region's height. Each run of such columns is split around its fill peaks, and for a color expected *n* times the *n* pieces with the highest fill are kept as wires. Partial blobs such as the black rubber boot or the foam sleeve get rejected this way. Two neighbouring wires of the same color need a visible gap or shadow between them to be counted separately. The wire centers are sorted by x and compared with `expected_order`.

Before that, every region's anchor is searched for, inside the region's search area if it has one, otherwise over the whole frame. The wires box is kept inside the search area. Without search areas, connectors of the same type look alike, so the matches are assigned to regions in the order the regions were configured, left to right (or top to bottom for stacked regions), and each box moves with its connector. A connector doesn't always look like its picture: the same part may come with or without its rubber boot. So in a region with a search area, if the connector isn't found, or the wires box it gives doesn't hold every expected wire, the app also slides the wires box over the whole area to where the expected colors run through it, and uses whichever placement finds more of the expected wires. Wires found this way must cover at least `detection.wire_search_min_fill` (60%) of the box height, so foam or a boot next to a missing wire can't stand in for it. Without a search area there is no such search, and a connector that can't be found fails with *Connector not found*. Rotated connectors, or a camera much closer or further away than in the reference image, won't match.

The settings you can tune in `config.json`:

- `colors.<name>.hsv_ranges`: OpenCV HSV bounds (H 0–180). Add new wire colors here.
- `colors.<name>.achromatic`: `true` for black and white. A pixel counts as *colored* if its saturation is at least `detection.colored_min_saturation` (110) or its chroma, the brightest minus the darkest RGB channel, is at least `detection.colored_min_chroma` (40). Achromatic colors only match pixels that aren't colored, the other colors only pixels that are. This keeps a black wire with a bluish camera tint from being read as blue, and a blue wire in shadow from being read as black.
- `colors.black.relative_dark`: a pixel also counts as black if it's at most this fraction (0.5) of the region's background brightness, so a black wire under bright light or high exposure still counts as black.
- `detection.min_column_fill`: fraction of the region height a column must cover to count as wire.
- `detection.anchor_min_score`: minimum template match (0–1) to accept the connector's new position.
- `profiles.<model>.image_size`: the resolution the boxes were drawn at. Boxes scale automatically to other resolutions.

## Project layout

- `wire_engine/`: the core logic: wire color detection, connector location and the order check. It has no GUI code, so it can be used from scripts:

  ```python
  import cv2
  from wire_engine import inspect, load_config

  result = inspect(cv2.imread("image.png"), load_config("config.json"), "FCFK2FF3")
  print(result.passed, [r.message for r in result.regions])
  ```

- `wire_inspector/`: the Tkinter app (`gui.py`) and camera handling (`camera.py`).
- `main.py`: starts the app.

## Tests

```
python -m pytest
```
