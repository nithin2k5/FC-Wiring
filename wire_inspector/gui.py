"""Tkinter frontend for wire color / order inspection."""
from __future__ import annotations

import copy
import csv
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .camera import RESOLUTIONS, list_cameras, open_camera
from wire_engine import draw_result, inspect, load_config, make_anchor, save_config, scale_box

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PASS_BG = "#1f9d3a"
FAIL_BG = "#c62828"
IDLE_BG = "#616161"


class WireInspectorApp:
    def __init__(self, root: tk.Tk, config_path: Path):
        self.root = root
        self.config_path = config_path
        self.config = load_config(config_path)

        self.frame = None          # current BGR frame
        self.raw_frame = None      # latest camera frame before flipping
        self.display = None        # BGR image currently shown (with overlay)
        self.photo = None          # keeps the Tk image alive
        self.scale = 1.0           # display px per frame px
        self.offset = (0, 0)       # canvas offset of the displayed image
        self.capture = None
        self.cameras = []
        self.camera_session = 0    # invalidates stale camera loops after restart
        self.read_failures = 0
        self.drag_start = None
        self.drag_rect = None
        # (region, "connector" | "area", chain) waiting for a drag; chain moves on to
        # the area step after the connector, when a new region is being set up.
        self.setup_step = None
        self.undo_stack = []       # (profile name, regions, image_size) before each region edit
        self.image_paths: list[Path] = []
        self.image_index = -1

        root.title("FC Wiring Inspector")
        root.geometry("1280x760")
        root.minsize(900, 560)
        self._build_ui()
        self._refresh_profiles()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<Left>", lambda e: self.step_image(-1))
        root.bind("<Right>", lambda e: self.step_image(1))
        root.bind("<Escape>", lambda e: self._cancel_step())
        root.bind("<Control-z>", lambda e: self.undo())
        self.refresh_cameras()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Open Images", command=self.open_images).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Open Folder", command=self.open_folder).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(toolbar, text="Profile").pack(side=tk.LEFT)
        self.profile_var = tk.StringVar()
        self.profile_box = ttk.Combobox(toolbar, textvariable=self.profile_var, state="readonly", width=18)
        self.profile_box.pack(side=tk.LEFT, padx=4)
        self.profile_box.bind("<<ComboboxSelected>>", lambda e: self._on_profile_change())
        ttk.Button(toolbar, text="New", command=self.new_profile).pack(side=tk.LEFT)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self.edit_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(toolbar, text="Edit regions (drag to add)", variable=self.edit_mode,
                        command=self._on_edit_toggle).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="Save Config", command=self.save).pack(side=tk.LEFT, padx=6)

        ttk.Button(toolbar, text="INSPECT", command=self.run_inspection).pack(side=tk.RIGHT)
        ttk.Button(toolbar, text="Save Result", command=self.save_result).pack(side=tk.RIGHT, padx=6)

        cambar = ttk.Frame(self.root, padding=(6, 0, 6, 6))
        cambar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(cambar, text="Camera").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value="Searching...")
        self.camera_box = ttk.Combobox(cambar, textvariable=self.camera_var, state="readonly", width=42)
        self.camera_box.pack(side=tk.LEFT, padx=4)
        self.camera_box.bind("<<ComboboxSelected>>", lambda e: self._restart_camera_if_running())
        self.refresh_btn = ttk.Button(cambar, text="Refresh", command=self.refresh_cameras)
        self.refresh_btn.pack(side=tk.LEFT)
        ttk.Label(cambar, text="Resolution").pack(side=tk.LEFT, padx=(10, 2))
        self.resolution_var = tk.StringVar(value="1280x720")
        res_box = ttk.Combobox(cambar, textvariable=self.resolution_var, values=RESOLUTIONS,
                               state="readonly", width=10)
        res_box.pack(side=tk.LEFT)
        res_box.bind("<<ComboboxSelected>>", lambda e: self._restart_camera_if_running())
        self.cam_btn = ttk.Button(cambar, text="Start Camera", command=self.toggle_camera, state=tk.DISABLED)
        self.cam_btn.pack(side=tk.LEFT, padx=(10, 0))
        self.capture_btn = ttk.Button(cambar, text="Capture", command=self.capture_frame, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.live_inspect = tk.BooleanVar(value=True)
        ttk.Checkbutton(cambar, text="Live inspect", variable=self.live_inspect).pack(side=tk.LEFT, padx=6)
        self.flip_vertical = tk.BooleanVar(value=self.config.get("flip_vertical", False))
        self.flip_horizontal = tk.BooleanVar(value=self.config.get("flip_horizontal", False))
        ttk.Checkbutton(cambar, text="Flip up/down", variable=self.flip_vertical,
                        command=self._on_flip).pack(side=tk.LEFT)
        ttk.Checkbutton(cambar, text="Flip left/right", variable=self.flip_horizontal,
                        command=self._on_flip).pack(side=tk.LEFT, padx=(6, 0))
        self.cam_info = ttk.Label(cambar, text="", foreground="#666")
        self.cam_info.pack(side=tk.LEFT, padx=6)

        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, bg="#202020", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        side = ttk.Frame(body, padding=8, width=360)
        side.pack(side=tk.RIGHT, fill=tk.Y)
        side.pack_propagate(False)

        self.verdict = tk.Label(side, text="NO IMAGE", font=("Segoe UI", 28, "bold"),
                                fg="white", bg=IDLE_BG, height=2)
        self.verdict.pack(fill=tk.X)

        ttk.Label(side, text="Results", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        cols = ("expected", "detected", "result")
        self.results = ttk.Treeview(side, columns=cols, show="tree headings", height=6)
        self.results.heading("#0", text="Region")
        self.results.column("#0", width=110)
        for c, wdt in zip(cols, (80, 80, 60)):
            self.results.heading(c, text=c.capitalize())
            self.results.column(c, width=wdt, anchor=tk.CENTER)
        self.results.tag_configure("pass", foreground=PASS_BG)
        self.results.tag_configure("fail", foreground=FAIL_BG)
        self.results.pack(fill=tk.X)
        self.detail = ttk.Label(side, text="", wraplength=340, justify=tk.LEFT)
        self.detail.pack(fill=tk.X, pady=4)

        ttk.Label(side, text="Test images", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        nav = ttk.Frame(side)
        nav.pack(fill=tk.X)
        ttk.Button(nav, text="< Prev", width=8, command=lambda: self.step_image(-1)).pack(side=tk.LEFT)
        self.image_counter = ttk.Label(nav, text="0 / 0", anchor=tk.CENTER)
        self.image_counter.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(nav, text="Next >", width=8, command=lambda: self.step_image(1)).pack(side=tk.LEFT)
        self.image_name = ttk.Label(side, text="", foreground="#666")
        self.image_name.pack(anchor=tk.W)
        ttk.Button(side, text="Test All Images", command=self.test_all).pack(fill=tk.X, pady=(4, 0))

        ttk.Label(side, text="Configured regions", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))
        self.region_list = tk.Listbox(side, height=6)
        self.region_list.pack(fill=tk.X)
        btns = ttk.Frame(side)
        btns.pack(fill=tk.X, pady=4)
        ttk.Button(btns, text="Edit order", command=self.edit_region_order).pack(side=tk.LEFT)
        ttk.Button(btns, text="Rename", command=self.rename_region).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Delete", command=self.delete_region).pack(side=tk.LEFT)
        ttk.Button(btns, text="Set area", command=lambda: self.set_region_box("area")).pack(side=tk.LEFT, padx=4)

        ttk.Label(side, text="Colors: " + ", ".join(self.config["colors"]),
                  foreground="#666").pack(anchor=tk.W, pady=(8, 0))

        self.status = ttk.Label(self.root, text="Open an image or start the camera.", relief=tk.SUNKEN,
                                anchor=tk.W, padding=(6, 2))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # -------------------------------------------------------------- profiles
    @property
    def profile(self) -> dict | None:
        return self.config["profiles"].get(self.profile_var.get())

    def _refresh_profiles(self, select: str | None = None):
        names = list(self.config["profiles"])
        self.profile_box["values"] = names
        if select or self.profile_var.get() not in names:
            self.profile_var.set(select or (names[0] if names else ""))
        self._on_profile_change()

    def _on_profile_change(self):
        self.region_list.delete(0, tk.END)
        if self.profile:
            for r in self.profile["regions"]:
                tag = ("  (area)" if r.get("search_box") else "") + ("" if r.get("anchor") else "  (fixed position)")
                self.region_list.insert(tk.END, f"{r['name']}:  {' | '.join(r['expected_order'])}{tag}")
        self._clear_results()
        self.redraw()

    def new_profile(self):
        name = simpledialog.askstring("New profile", "Profile / model name:", parent=self.root)
        if not name:
            return
        if name in self.config["profiles"]:
            messagebox.showerror("New profile", f"Profile '{name}' already exists.")
            return
        size = [self.frame.shape[1], self.frame.shape[0]] if self.frame is not None else None
        self.config["profiles"][name] = {"description": "", "image_size": size, "regions": []}
        self._refresh_profiles(select=name)
        self.edit_mode.set(True)
        self.set_status(f"Profile '{name}' created. Drag on the image to add regions.")

    def save(self):
        save_config(self.config, self.config_path)
        self.set_status(f"Saved configuration to {self.config_path}")

    # ------------------------------------------------------------- regions
    def _selected_region(self):
        sel = self.region_list.curselection()
        if not sel or not self.profile:
            messagebox.showinfo("Regions", "Select a region in the list first.")
            return None
        return self.profile["regions"][sel[0]]

    def _ask_order(self, initial: str):
        colors = self.config["colors"]
        text = simpledialog.askstring(
            "Expected order", f"Wire colors from LEFT to RIGHT, comma separated.\nAvailable: {', '.join(colors)}",
            initialvalue=initial, parent=self.root)
        if text is None:
            return None
        order = [c.strip().lower() for c in text.split(",") if c.strip()]
        unknown = [c for c in order if c not in colors]
        if not order or unknown:
            messagebox.showerror("Expected order", "Unknown colors: " + ", ".join(unknown) if unknown
                                 else "Enter at least one color.")
            return None
        return order

    def edit_region_order(self):
        region = self._selected_region()
        if region:
            order = self._ask_order(", ".join(region["expected_order"]))
            if order:
                self._push_undo()
                region["expected_order"] = order
                self._on_profile_change()

    def rename_region(self):
        region = self._selected_region()
        if region:
            name = simpledialog.askstring("Rename", "Region name:", initialvalue=region["name"], parent=self.root)
            if name:
                self._push_undo()
                region["name"] = name
                self._on_profile_change()

    def set_region_box(self, kind):
        region = self._selected_region()
        if region and self.frame is not None:
            self._ask_step(region, kind)

    def _ask_step(self, region, kind, chain=False):
        self.setup_step = (region, kind, chain)
        self.edit_mode.set(True)
        self.redraw()
        if kind == "connector":
            self.set_status(f"Drag a tight box around the connector body of '{region['name']}' "
                            "(housing and boot, not the wires). Esc to skip.")
        else:
            self.set_status(f"Drag the search area for '{region['name']}': a box that holds this connector "
                            "and its wires wherever they can be, and no other connector. Esc to skip.")

    def _cancel_step(self):
        if self.setup_step is None:
            return
        region, kind, chain = self.setup_step
        self.setup_step = None
        if kind == "connector" and chain:
            self._ask_step(region, "area")
            return
        if region.get("search_box"):
            how = "is only looked for inside its search area"
        elif region.get("anchor"):
            how = "is looked for anywhere in the frame"
        else:
            how = "stays at a fixed position"
        self.set_status(f"Skipped. Region '{region['name']}' {how}.")

    def _push_undo(self):
        profile = self.profile
        self.undo_stack.append((self.profile_var.get(), copy.deepcopy(profile["regions"]),
                                copy.deepcopy(profile.get("image_size"))))
        del self.undo_stack[:-30]

    def undo(self):
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry)):
            return
        self.setup_step = None
        if not self.undo_stack:
            self.set_status("Nothing to undo.")
            self.redraw()
            return
        name, regions, size = self.undo_stack.pop()
        profile = self.config["profiles"].get(name)
        if profile is None:
            self.set_status(f"Can't undo: profile '{name}' no longer exists.")
            return
        profile["regions"], profile["image_size"] = regions, size
        if self.profile_var.get() != name:
            self.profile_var.set(name)
        self._on_profile_change()
        self.set_status("Undone. Remember to Save Config.")

    def _on_edit_toggle(self):
        if self.edit_mode.get():
            self.redraw()
        else:
            self.setup_step = None
            self.run_inspection(quiet=True)

    def start_detection(self):
        """Leave edit mode and inspect: live if the camera runs, else the current image."""
        self.setup_step = None
        self.edit_mode.set(False)
        if self.capture is not None:
            self.live_inspect.set(True)
            self.set_status("Detection running on the live camera.")
        else:
            self.run_inspection(quiet=True)

    def delete_region(self):
        region = self._selected_region()
        if region and messagebox.askyesno("Delete", f"Delete region '{region['name']}'?"):
            self._push_undo()
            self.profile["regions"].remove(region)
            self._on_profile_change()

    # --------------------------------------------------------------- mouse
    def _canvas_to_frame(self, x, y):
        fx = (x - self.offset[0]) / self.scale
        fy = (y - self.offset[1]) / self.scale
        h, w = self.frame.shape[:2]
        return min(max(fx, 0), w), min(max(fy, 0), h)

    def on_press(self, event):
        if ((self.edit_mode.get() or self.setup_step is not None)
                and self.frame is not None and self.profile is not None):
            self.drag_start = (event.x, event.y)

    def on_drag(self, event):
        if not self.drag_start:
            return
        if self.drag_rect:
            self.canvas.delete(self.drag_rect)
        self.drag_rect = self.canvas.create_rectangle(*self.drag_start, event.x, event.y,
                                                      outline="#ffeb3b", width=2, dash=(4, 2))

    def on_release(self, event):
        if not self.drag_start:
            return
        start, self.drag_start = self.drag_start, None
        if self.drag_rect:
            self.canvas.delete(self.drag_rect)
            self.drag_rect = None
        x1, y1 = self._canvas_to_frame(*start)
        x2, y2 = self._canvas_to_frame(event.x, event.y)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 10 or y2 - y1 < 10:
            return

        profile = self.profile
        h, w = self.frame.shape[:2]
        if not profile.get("image_size"):
            profile["image_size"] = [w, h]
        # Store the box in the profile's reference resolution.
        pw, ph = profile["image_size"]
        box = [round(x1 * pw / w), round(y1 * ph / h), round(x2 * pw / w), round(y2 * ph / h)]

        if self.setup_step is not None:
            region, kind, chain = self.setup_step
            self.setup_step = None
            if kind == "connector":
                anchor = make_anchor(self.frame, box, profile["image_size"])
                if anchor:
                    self._push_undo()
                    region["anchor"] = anchor
                self._on_profile_change()
                if chain:
                    self._ask_step(region, "area")
                    return
                self.set_status(f"Connector set for '{region['name']}'. Remember to Save Config.")
            else:
                self._push_undo()
                region["search_box"] = box
                self._on_profile_change()
                inside = all(self._contains(box, b) for b in
                             [region["box"]] + ([region["anchor"]["box"]] if region.get("anchor") else []))
                note = "" if inside else (" Note: the area doesn't fully contain the wires and connector "
                                          "boxes, so they may be cut off.")
                self.set_status(f"Search area set for '{region['name']}'.{note} Remember to Save Config.")
                if messagebox.askyesno("Start detection",
                                       f"Search area set for '{region['name']}'.{note}\n\nStart detection now?\n"
                                       "Choose No to set up another connector first.", parent=self.root):
                    self.start_detection()
            return

        name = simpledialog.askstring("New region", "Region name:",
                                      initialvalue=f"Region {len(profile['regions']) + 1}", parent=self.root)
        if not name:
            self.redraw()
            return
        order = self._ask_order("black, blue")
        if not order:
            self.redraw()
            return
        region = {"name": name, "box": box, "expected_order": order}
        self._push_undo()
        profile["regions"].append(region)
        self._on_profile_change()
        self._ask_step(region, "connector", chain=True)

    @staticmethod
    def _contains(outer, inner):
        return (outer[0] <= inner[0] and outer[1] <= inner[1]
                and outer[2] >= inner[2] and outer[3] >= inner[3])

    # --------------------------------------------------------------- input
    @staticmethod
    def read_image(path: Path):
        # cv2.imread can't open non-ASCII paths on Windows; decode from bytes instead.
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None

    def _orient(self, frame):
        """Apply the flip settings to an incoming image or camera frame."""
        v, h = self.flip_vertical.get(), self.flip_horizontal.get()
        if v and h:
            return cv2.flip(frame, -1)
        if v or h:
            return cv2.flip(frame, 0 if v else 1)
        return frame

    def _on_flip(self):
        self.config["flip_vertical"] = self.flip_vertical.get()
        self.config["flip_horizontal"] = self.flip_horizontal.get()
        if self.capture is None and 0 <= self.image_index < len(self.image_paths):
            self.show_image(self.image_index)
        flips = [name for name, on in (("up/down", self.config["flip_vertical"]),
                                       ("left/right", self.config["flip_horizontal"])) if on]
        state = "Images are flipped " + " and ".join(flips) if flips else "Images are not flipped"
        self.set_status(state + ". Save Config to keep this setting.")

    def open_images(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if paths:
            self._set_images([Path(p) for p in paths])

    def open_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not paths:
            messagebox.showinfo("Open folder", "No images found in that folder.")
            return
        self._set_images(paths)

    def _set_images(self, paths: list[Path]):
        self.stop_camera()
        self.image_paths = paths
        self.show_image(0)

    def step_image(self, delta: int):
        if self.image_paths and not isinstance(self.root.focus_get(), (tk.Entry, ttk.Entry, tk.Listbox)):
            self.show_image((self.image_index + delta) % len(self.image_paths))

    def show_image(self, index: int):
        self.stop_camera()
        self.image_index = index
        path = self.image_paths[index]
        self.image_counter.config(text=f"{index + 1} / {len(self.image_paths)}")
        self.image_name.config(text=path.name)
        frame = self.read_image(path)
        if frame is None:
            self.frame = self.display = None
            self._clear_results("UNREADABLE")
            self.redraw()
            self.set_status(f"Could not read {path}")
            return
        self.frame = self._orient(frame)
        self.set_status(f"Loaded {path.name}  ({frame.shape[1]}x{frame.shape[0]})")
        self.run_inspection()

    def test_all(self):
        if not self.image_paths:
            messagebox.showinfo("Test all", "Open images or a folder first.")
            return
        if not self.profile or not self.profile["regions"]:
            messagebox.showinfo("Test all", "The selected profile has no regions.")
            return
        rows = []
        for path in self.image_paths:
            frame = self.read_image(path)
            if frame is None:
                rows.append((path, "ERROR", "Could not read image"))
                continue
            result = inspect(self._orient(frame), self.config, self.profile_var.get())
            rows.append((path, "PASS" if result.passed else "FAIL",
                         "; ".join(f"{r.name}: {r.message}" for r in result.regions)))
        BatchReport(self, rows, self.profile_var.get())

    def refresh_cameras(self):
        if self.capture:
            self.stop_camera()
        self.refresh_btn.config(state=tk.DISABLED)
        self.cam_btn.config(state=tk.DISABLED)
        self.camera_box["values"] = []
        self.camera_var.set("Searching for cameras...")

        # Probing takes a few seconds; do it off the UI thread. Tk calls must stay
        # on the main thread, so poll for the result instead of calling back.
        result = []
        thread = threading.Thread(target=lambda: result.append(list_cameras()), daemon=True)
        thread.start()

        def poll():
            if thread.is_alive():
                self.root.after(100, poll)
            else:
                self._cameras_found(result[0] if result else [])

        poll()

    def _cameras_found(self, cams):
        previous = self._selected_camera()
        self.cameras = cams
        self.refresh_btn.config(state=tk.NORMAL)
        if not cams:
            self.camera_var.set("No camera found")
            self.set_status("No camera found. Connect one and press Refresh.")
            return
        self.camera_box["values"] = [c.label for c in cams]
        keep = next((c for c in cams if previous and c.index == previous.index), cams[0])
        self.camera_var.set(keep.label)
        self.cam_btn.config(state=tk.NORMAL)
        self.set_status(f"Found {len(cams)} camera(s).")

    def _selected_camera(self):
        return next((c for c in self.cameras if c.label == self.camera_var.get()), None)

    def _restart_camera_if_running(self):
        if self.capture:
            self.stop_camera()
            self.start_camera()

    def toggle_camera(self):
        if self.capture:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        cam = self._selected_camera()
        if cam is None:
            return
        self.set_status(f"Opening {cam.name}...")
        self.root.update_idletasks()
        cap = open_camera(cam.index, self.resolution_var.get())
        if cap is None:
            messagebox.showerror("Camera", f"Could not open {cam.name}. It may be in use by another app.")
            return
        self.capture = cap
        self.camera_session += 1
        self.read_failures = 0
        self.cam_btn.config(text="Stop Camera")
        self.capture_btn.config(state=tk.NORMAL)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cam_info.config(text=f"Live {w}x{h}")
        self.set_status(f"{cam.name} running at {w}x{h}.")
        self._camera_loop(self.camera_session)

    def stop_camera(self):
        if self.capture:
            self.capture.release()
            self.capture = None
            self.cam_btn.config(text="Start Camera")
            self.capture_btn.config(state=tk.DISABLED)
            self.cam_info.config(text="")
            self.set_status("Camera stopped. Last frame kept.")

    def _camera_loop(self, session):
        if not self.capture or session != self.camera_session:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.read_failures += 1
            if self.read_failures >= 30:
                self.stop_camera()
                messagebox.showerror("Camera", "Camera stopped sending frames (disconnected?).")
                return
        else:
            self.read_failures = 0
            self.raw_frame = frame
            self.frame = self._orient(frame)
            if self.live_inspect.get() and not self.edit_mode.get():
                self.run_inspection(quiet=True)
            else:
                self.display = self.frame
                self.redraw()
        self.root.after(60, self._camera_loop, session)

    def capture_frame(self):
        """Freeze the current camera frame, save it to captures/ and inspect it."""
        if self.capture is None or self.raw_frame is None:
            return
        raw = self.raw_frame
        self.stop_camera()
        folder = self.config_path.parent / "captures"
        name = f"{self.profile_var.get() or 'capture'}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = folder / name
        try:
            folder.mkdir(exist_ok=True)
            ok, png = cv2.imencode(".png", raw)
            if not ok:
                raise OSError("could not encode image")
            png.tofile(str(path))  # cv2.imwrite can't handle non-ASCII paths on Windows
        except OSError as e:
            # Still inspect the frame even if it couldn't be saved.
            self.frame = self._orient(raw)
            self.run_inspection()
            self.set_status(f"Captured frame inspected, but saving it failed: {e}")
            return
        self.image_paths.append(path)
        self.show_image(len(self.image_paths) - 1)
        self.set_status(f"Captured {path}. Press Start Camera to go live again.")

    # ----------------------------------------------------------- inspection
    def run_inspection(self, quiet=False):
        if self.frame is None:
            if not quiet:
                messagebox.showinfo("Inspect", "Open an image or start the camera first.")
            return
        if not self.profile or not self.profile["regions"]:
            self.display = self.frame
            self._clear_results("NO REGIONS")
            self.redraw()
            return
        result = inspect(self.frame, self.config, self.profile_var.get())
        self.last_result = result
        self.display = draw_result(self.frame, result)
        self._show_result(result)
        self.redraw()

    def _clear_results(self, text="READY"):
        self.results.delete(*self.results.get_children())
        self.detail.config(text="")
        self.verdict.config(text=text, bg=IDLE_BG)
        self.last_result = None

    def _show_result(self, result):
        self.results.delete(*self.results.get_children())
        for r in result.regions:
            tag = "pass" if r.passed else "fail"
            self.results.insert("", tk.END, text=r.name, tags=(tag,), values=(
                " | ".join(r.expected_order), " | ".join(r.detected_order) or "-",
                "PASS" if r.passed else "FAIL"))
        self.detail.config(text="\n".join(f"{r.name}: {r.message}" for r in result.regions))
        self.verdict.config(text="PASS" if result.passed else "FAIL",
                            bg=PASS_BG if result.passed else FAIL_BG)

    def save_result(self):
        if self.display is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            cv2.imwrite(path, self.display)
            self.set_status(f"Saved {path}")

    # -------------------------------------------------------------- render
    def redraw(self):
        self.canvas.delete("all")
        base = self.display if self.display is not None else self.frame
        if base is None:
            return
        img = base.copy()
        if self.edit_mode.get() and self.profile:
            h, w = img.shape[:2]
            for r in self.profile["regions"]:
                x1, y1, x2, y2 = scale_box(r["box"], self.profile.get("image_size"), (w, h))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 235, 255), 2)
                cv2.putText(img, r["name"], (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 235, 255), 1, cv2.LINE_AA)
                if r.get("anchor"):
                    ax1, ay1, ax2, ay2 = scale_box(r["anchor"]["box"], self.profile.get("image_size"), (w, h))
                    cv2.rectangle(img, (ax1, ay1), (ax2, ay2), (255, 0, 255), 1)
                if r.get("search_box"):
                    sx1, sy1, sx2, sy2 = scale_box(r["search_box"], self.profile.get("image_size"), (w, h))
                    cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (255, 200, 0), 1)
                    cv2.putText(img, f"{r['name']} area", (sx1 + 3, sy1 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (255, 200, 0), 1, cv2.LINE_AA)

        cw, ch = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
        h, w = img.shape[:2]
        self.scale = min(cw / w, ch / h)
        dw, dh = max(int(w * self.scale), 1), max(int(h * self.scale), 1)
        self.offset = ((cw - dw) // 2, (ch - dh) // 2)
        rgb = cv2.cvtColor(cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(*self.offset, image=self.photo, anchor=tk.NW)

    def set_status(self, text):
        self.status.config(text=text)

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


class BatchReport(tk.Toplevel):
    """Summary window for testing every loaded image against a profile."""

    def __init__(self, app: WireInspectorApp, rows, profile: str):
        super().__init__(app.root)
        self.app = app
        self.rows = rows
        self.profile = profile
        passed = sum(1 for _, status, _ in rows if status == "PASS")
        self.title(f"Batch results - {profile}")
        self.geometry("900x420")

        tk.Label(self, text=f"{passed} / {len(rows)} passed    {len(rows) - passed} failed",
                 font=("Segoe UI", 14, "bold"), fg="white",
                 bg=PASS_BG if passed == len(rows) else FAIL_BG, pady=6).pack(fill=tk.X)

        frame = ttk.Frame(self, padding=6)
        frame.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(frame, columns=("result", "details"), show="tree headings")
        tree.heading("#0", text="Image")
        tree.column("#0", width=220)
        tree.heading("result", text="Result")
        tree.column("result", width=70, anchor=tk.CENTER)
        tree.heading("details", text="Details")
        tree.column("details", width=560)
        tree.tag_configure("pass", foreground=PASS_BG)
        tree.tag_configure("fail", foreground=FAIL_BG)
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        for i, (path, status, details) in enumerate(rows):
            tree.insert("", tk.END, iid=str(i), text=path.name, values=(status, details),
                        tags=("pass" if status == "PASS" else "fail",))
        tree.bind("<Double-1>", self._open_selected)
        self.tree = tree

        bar = ttk.Frame(self, padding=6)
        bar.pack(fill=tk.X)
        ttk.Label(bar, text="Double-click a row to view it.", foreground="#666").pack(side=tk.LEFT)
        ttk.Button(bar, text="Export CSV", command=self.export_csv).pack(side=tk.RIGHT)

    def _open_selected(self, _event):
        sel = self.tree.selection()
        if sel:
            self.app.show_image(int(sel[0]))

    def export_csv(self):
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".csv",
                                            initialfile=f"{self.profile}_results.csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "profile", "result", "details"])
            for img, status, details in self.rows:
                writer.writerow([str(img), self.profile, status, details])
        self.app.set_status(f"Exported {path}")


def run(config_path: Path):
    root = tk.Tk()
    WireInspectorApp(root, config_path)
    root.mainloop()
