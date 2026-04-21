from __future__ import annotations

import base64
import copy
import json
import queue
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from photo_reviver.app_utils import (
    backend_readiness,
    describe_analysis,
    describe_decision,
    describe_preprocess,
    describe_restoration,
    describe_validation,
)
from photo_reviver.colorization import (
    PALETTE_PRESETS,
    apply_palette_colorization,
    apply_staged_colorization,
    available_palette_presets,
    colorization_assets_ready,
    recommend_palette_key,
)
from photo_reviver.config import apply_cli_overrides, load_config
from photo_reviver.evaluate import create_comparison_grid
from photo_reviver.io_utils import load_image, save_image, save_json
from photo_reviver.pipeline import run_pipeline
from photo_reviver.postprocess import (
    apply_enhancement_controls,
    build_recommended_enhancement_settings,
    describe_enhancement_recommendation,
)


IMAGE_TYPES = [
    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
    ("All files", "*.*"),
]


PROCESS_STEPS = [
    ("upload", "Upload"),
    ("analysis", "Analysis"),
    ("preprocess", "Preprocess"),
    ("restoration", "Restoration"),
    ("colorization", "Colorization"),
    ("enhancement", "Enhancement"),
    ("comparison", "Comparison"),
]


SLIDER_SPECS = [
    ("brightness", "Brightness", -50, 50),
    ("contrast", "Contrast", -50, 80),
    ("gamma", "Gamma", -50, 50),
    ("saturation", "Saturation", -50, 80),
    ("vibrance", "Vibrance", -50, 90),
    ("warmth", "Warmth", -50, 50),
    ("tint", "Tint", -50, 50),
    ("clarity", "Clarity", 0, 80),
    ("denoise", "Denoise", 0, 80),
    ("sharpness", "Sharpness", 0, 80),
]


def build_restoration_only_config(config: dict) -> dict:
    run_config = copy.deepcopy(config)
    run_config["postprocess"]["apply_enhancement"] = False
    run_config["postprocess"]["enhancement_strength"] = 0.0
    run_config["postprocess"]["apply_sharpening"] = False
    run_config["postprocess"]["sharpening_strength"] = 0.0
    run_config["postprocess"]["simple_upscale_factor"] = 1.0
    run_config["postprocess"]["attempt_colorization"] = False
    run_config["postprocess"]["colorization"]["enabled"] = False
    return run_config


def image_to_photo(path: Path, max_width: int, max_height: int) -> tk.PhotoImage:
    image = load_image(path)
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"Could not prepare image preview: {path}")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return tk.PhotoImage(data=payload)


def slug_part(value: str | None) -> str:
    if not value:
        return "none"
    cleaned = "".join(character if character.isalnum() else "-" for character in value.lower())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "none"


class PhotoReviverDesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Photo Reviver")
        self.root.geometry("1280x820")
        self.root.minsize(1100, 720)

        self.config = load_config()
        boptl_ready, _ = backend_readiness(self.config["restoration"])
        self.backend_var = tk.StringVar(value="boptl" if boptl_ready else "passthrough")
        self.gpu_var = tk.StringVar(value=str(self.config["restoration"].get("gpu", "-1")))
        self.max_side_var = tk.IntVar(
            value=int(self.config["preprocess"].get("model_safe_resize_longest_side") or 768)
        )
        self.status_var = tk.StringVar(value="Upload a photo to begin.")
        self.palette_var = tk.StringVar()
        self.palette_note_var = tk.StringVar(value="")
        self.recommended_palette_var = tk.StringVar(value="")
        self.intensity_var = tk.DoubleVar(value=0.85)
        self.deoldify_note_var = tk.StringVar(value="")

        self.uploaded_path: Path | None = None
        self.summary: dict | None = None
        self.original_path: Path | None = None
        self.restored_path: Path | None = None
        self.colorized_path: Path | None = None
        self.pending_colorized_path: Path | None = None
        self.pending_color_base_path: Path | None = None
        self.pending_color_version_dir: Path | None = None
        self.colorization_version_index = 0
        self.enhancement_base_path: Path | None = None
        self.enhanced_path: Path | None = None
        self.comparison_path: Path | None = None
        self.enhancement_version = 0

        self.event_queue: queue.Queue = queue.Queue()
        self.photo_refs: dict[str, tk.PhotoImage] = {}
        self.step_labels: dict[str, ttk.Label] = {}
        self.slider_vars: dict[str, tk.DoubleVar] = {}
        self.slider_value_labels: dict[str, ttk.Label] = {}

        self.configure_style()
        self.build_layout()
        self.populate_palette_options()
        self.set_initial_control_state()
        self.root.after(150, self.drain_events)

    def configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f5f2ec")
        style.configure("Sidebar.TFrame", background="#ebe6dc")
        style.configure("TLabel", background="#f5f2ec", foreground="#292724")
        style.configure("Sidebar.TLabel", background="#ebe6dc", foreground="#292724")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Stage.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", padding=(10, 6))
        style.configure("Accent.TButton", padding=(12, 8), font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe", background="#f5f2ec")
        style.configure("TLabelframe.Label", background="#f5f2ec", foreground="#292724")

    def build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", padding=16)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.columnconfigure(0, weight=1)

        ttk.Label(sidebar, text="Photo Reviver", style="Title.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
        )
        ttk.Label(
            sidebar,
            textvariable=self.status_var,
            style="Sidebar.TLabel",
            wraplength=230,
        ).grid(row=1, column=0, sticky="ew", pady=(8, 18))

        controls = ttk.LabelFrame(sidebar, text="Run", padding=10)
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Restoration engine").grid(row=0, column=0, sticky="w")
        self.backend_combo = ttk.Combobox(
            controls,
            textvariable=self.backend_var,
            values=["passthrough", "boptl"],
            state="readonly",
        )
        self.backend_combo.grid(row=1, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(controls, text="GPU ids").grid(row=2, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.gpu_var).grid(row=3, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(controls, text="BOPTL max side").grid(row=4, column=0, sticky="w")
        ttk.Spinbox(
            controls,
            from_=384,
            to=1600,
            increment=64,
            textvariable=self.max_side_var,
        ).grid(row=5, column=0, sticky="ew", pady=(2, 10))

        self.upload_button = ttk.Button(
            controls,
            text="Upload Photo",
            command=self.upload_photo,
        )
        self.upload_button.grid(row=6, column=0, sticky="ew", pady=(0, 8))

        self.restore_button = ttk.Button(
            controls,
            text="Restore Photo",
            style="Accent.TButton",
            command=self.start_restoration,
        )
        self.restore_button.grid(row=7, column=0, sticky="ew")

        process_box = ttk.LabelFrame(sidebar, text="Process", padding=10)
        process_box.grid(row=3, column=0, sticky="ew", pady=(16, 0))
        process_box.columnconfigure(0, weight=1)
        for row, (key, label) in enumerate(PROCESS_STEPS):
            status = ttk.Label(
                process_box,
                text=f"Waiting - {label}",
                style="Sidebar.TLabel",
            )
            status.grid(row=row, column=0, sticky="w", pady=2)
            self.step_labels[key] = status

        self.download_button = ttk.Button(
            sidebar,
            text="Download Final Product",
            command=self.export_final_product,
        )
        self.download_button.grid(row=4, column=0, sticky="ew", pady=(18, 0))

        main = ttk.Frame(self.root, padding=14)
        main.grid(row=0, column=1, sticky="nsew")
        main.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.build_upload_tab()
        self.build_restoration_tab()
        self.build_colorization_tab()
        self.build_enhancement_tab()
        self.build_comparison_tab()

    def build_upload_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=16)
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)
        self.notebook.add(tab, text="Upload")

        ttk.Label(tab, text="Uploaded photo", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.upload_preview = ttk.Label(tab, anchor="center")
        self.upload_preview.grid(row=1, column=0, sticky="nsew", pady=12)
        self.upload_path_label = ttk.Label(tab, text="No photo selected.", wraplength=820)
        self.upload_path_label.grid(row=2, column=0, sticky="w")

    def build_restoration_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=16)
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)
        self.notebook.add(tab, text="Restoration")

        ttk.Label(tab, text="Restoration process", style="Title.TLabel").grid(row=0, column=0, sticky="w")

        images = ttk.Frame(tab)
        images.grid(row=1, column=0, sticky="nsew", pady=12)
        for column in range(3):
            images.columnconfigure(column, weight=1)
        self.original_preview = self.add_image_panel(images, "Original", 0)
        self.preprocess_preview = self.add_image_panel(images, "Preprocessed", 1)
        self.restored_preview = self.add_image_panel(images, "After restoration", 2)

        self.restoration_text = tk.Text(tab, height=10, wrap="word", relief="flat", padx=10, pady=10)
        self.restoration_text.grid(row=2, column=0, sticky="ew")
        self.restoration_text.configure(state="disabled")

        self.to_colorization_button = ttk.Button(
            tab,
            text="Move to Colorization",
            command=self.move_to_colorization,
        )
        self.to_colorization_button.grid(row=3, column=0, sticky="e", pady=(12, 0))

    def build_colorization_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=16)
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)
        self.notebook.add(tab, text="Colorization")

        ttk.Label(tab, text="Colorization", style="Title.TLabel").grid(row=0, column=0, sticky="w")

        body = ttk.Frame(tab)
        body.grid(row=1, column=0, sticky="nsew", pady=12)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        preview = ttk.Frame(body)
        preview.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        preview.columnconfigure(0, weight=1)
        preview.columnconfigure(1, weight=1)
        self.color_before_preview = self.add_image_panel(preview, "After restoration", 0)
        self.color_after_preview = self.add_image_panel(preview, "After colorization", 1)

        controls = ttk.LabelFrame(body, text="Colorization", padding=12)
        controls.grid(row=0, column=1, sticky="nsew")
        controls.columnconfigure(0, weight=1)

        ttk.Label(
            controls,
            text="1. Run DeOldify model",
            style="Stage.TLabel",
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(controls, textvariable=self.deoldify_note_var, wraplength=360).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(6, 10),
        )

        self.colorize_button = ttk.Button(
            controls,
            text="Run Model",
            style="Accent.TButton",
            command=self.apply_colorization,
        )
        self.colorize_button.grid(row=2, column=0, sticky="ew", pady=(0, 14))

        ttk.Label(
            controls,
            text="2. Add palette after model",
            style="Stage.TLabel",
        ).grid(row=3, column=0, sticky="w")

        self.palette_combo = ttk.Combobox(
            controls,
            textvariable=self.palette_var,
            state="readonly",
        )
        self.palette_combo.grid(row=4, column=0, sticky="ew", pady=(8, 4))
        self.palette_combo.bind("<<ComboboxSelected>>", self.update_palette_note)

        ttk.Label(controls, textvariable=self.palette_note_var, wraplength=360).grid(
            row=5,
            column=0,
            sticky="ew",
            pady=(0, 6),
        )

        ttk.Label(controls, text="Palette intensity").grid(row=6, column=0, sticky="w")
        intensity = ttk.Scale(
            controls,
            from_=0.15,
            to=1.0,
            variable=self.intensity_var,
            command=lambda _value: self.update_intensity_label(),
        )
        intensity.grid(row=7, column=0, sticky="ew")
        self.intensity_label = ttk.Label(controls, text="")
        self.intensity_label.grid(row=8, column=0, sticky="w", pady=(2, 10))

        self.update_post_palette_button = ttk.Button(
            controls,
            text="Add After-Model Palette",
            command=self.update_after_palette_preview,
        )
        self.update_post_palette_button.grid(row=9, column=0, sticky="ew", pady=(0, 14))

        ttk.Label(
            controls,
            text="3. Choose what moves forward",
            style="Stage.TLabel",
        ).grid(row=10, column=0, sticky="w", pady=(0, 8))

        self.keep_color_button = ttk.Button(
            controls,
            text="Keep This Version",
            command=self.keep_colorization_preview,
        )
        self.keep_color_button.grid(row=11, column=0, sticky="ew", pady=(0, 8))

        self.keep_original_button = ttk.Button(
            controls,
            text="Keep Restored Original",
            command=self.keep_restored_original,
        )
        self.keep_original_button.grid(row=12, column=0, sticky="ew", pady=(0, 8))

        self.skip_color_button = ttk.Button(
            controls,
            text="Skip and Keep Original",
            command=self.skip_colorization,
        )
        self.skip_color_button.grid(row=13, column=0, sticky="ew", pady=(0, 8))

        self.to_enhancement_button = ttk.Button(
            controls,
            text="Move to Enhancement",
            command=self.move_to_enhancement,
        )
        self.to_enhancement_button.grid(row=14, column=0, sticky="ew")

    def build_enhancement_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=16)
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)
        self.notebook.add(tab, text="Enhancement")

        ttk.Label(tab, text="Enhancement", style="Title.TLabel").grid(row=0, column=0, sticky="w")

        body = ttk.Frame(tab)
        body.grid(row=1, column=0, sticky="nsew", pady=12)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)

        preview = ttk.Frame(body)
        preview.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        preview.columnconfigure(0, weight=1)
        preview.columnconfigure(1, weight=1)
        self.enhance_before_preview = self.add_image_panel(preview, "Input", 0)
        self.enhance_after_preview = self.add_image_panel(preview, "After final touches", 1)

        controls = ttk.LabelFrame(body, text="Controls", padding=12)
        controls.grid(row=0, column=1, sticky="ns")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, textvariable=self.recommended_palette_var, wraplength=340).grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(0, 10),
        )

        for row, (key, label, start, end) in enumerate(SLIDER_SPECS, start=1):
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.DoubleVar(value=0)
            self.slider_vars[key] = var
            slider = ttk.Scale(
                controls,
                from_=start,
                to=end,
                variable=var,
                command=lambda _value, item=key: self.update_slider_label(item),
            )
            slider.grid(row=row, column=1, sticky="ew", padx=8, pady=2)
            value_label = ttk.Label(controls, text="0", width=5)
            value_label.grid(row=row, column=2, sticky="e")
            self.slider_value_labels[key] = value_label

        button_row = len(SLIDER_SPECS) + 2
        ttk.Button(
            controls,
            text="Reset to Recommended",
            command=self.reset_enhancement_defaults,
        ).grid(row=button_row, column=0, columnspan=3, sticky="ew", pady=(12, 6))

        self.enhance_button = ttk.Button(
            controls,
            text="Enhance",
            style="Accent.TButton",
            command=lambda: self.apply_enhancement(use_current_output=False),
        )
        self.enhance_button.grid(row=button_row + 1, column=0, columnspan=3, sticky="ew", pady=(0, 6))

        self.enhance_again_button = ttk.Button(
            controls,
            text="Enhance Again",
            command=lambda: self.apply_enhancement(use_current_output=True),
        )
        self.enhance_again_button.grid(row=button_row + 2, column=0, columnspan=3, sticky="ew", pady=(0, 6))

        self.to_final_button = ttk.Button(
            controls,
            text="Move to Final View",
            command=self.move_to_final_view,
        )
        self.to_final_button.grid(row=button_row + 3, column=0, columnspan=3, sticky="ew")

    def build_comparison_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=16)
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)
        self.notebook.add(tab, text="Comparison")

        ttk.Label(tab, text="Stage comparison", style="Title.TLabel").grid(row=0, column=0, sticky="w")

        grid = ttk.Frame(tab)
        grid.grid(row=1, column=0, sticky="nsew", pady=12)
        for column in range(4):
            grid.columnconfigure(column, weight=1)
        self.compare_original_preview = self.add_image_panel(grid, "Original", 0)
        self.compare_restored_preview = self.add_image_panel(grid, "After restoration", 1)
        self.compare_color_preview = self.add_image_panel(grid, "After colorization", 2)
        self.compare_final_preview = self.add_image_panel(grid, "After final touches", 3)

        self.comparison_full_preview = ttk.Label(tab, anchor="center")
        self.comparison_full_preview.grid(row=2, column=0, sticky="nsew")

    def add_image_panel(self, parent: ttk.Frame, title: str, column: int) -> ttk.Label:
        panel = ttk.Frame(parent, padding=6)
        panel.grid(row=0, column=column, sticky="nsew")
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text=title, style="Stage.TLabel").grid(row=0, column=0, sticky="w")
        image_label = ttk.Label(panel, anchor="center")
        image_label.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        return image_label

    def populate_palette_options(self) -> None:
        self.palette_options = available_palette_presets()
        self.palette_labels = {
            f"{item['name']} - {item['description']}": item["key"]
            for item in self.palette_options
        }
        self.palette_combo.configure(values=list(self.palette_labels.keys()))
        first_label = next(iter(self.palette_labels))
        self.palette_var.set(first_label)
        self.update_palette_note()
        self.update_intensity_label()
        self.update_deoldify_note()

    def set_initial_control_state(self) -> None:
        self.reset_process_steps()
        self.restore_button.configure(state="disabled")
        self.download_button.configure(state="disabled")
        self.to_colorization_button.configure(state="disabled")
        self.colorize_button.configure(state="disabled")
        self.update_post_palette_button.configure(state="disabled")
        self.keep_color_button.configure(state="disabled")
        self.keep_original_button.configure(state="disabled")
        self.skip_color_button.configure(state="disabled")
        self.to_enhancement_button.configure(state="disabled")
        self.enhance_button.configure(state="disabled")
        self.enhance_again_button.configure(state="disabled")
        self.to_final_button.configure(state="disabled")

    def upload_photo(self) -> None:
        selected = filedialog.askopenfilename(title="Choose a photo", filetypes=IMAGE_TYPES)
        if not selected:
            return

        self.uploaded_path = Path(selected)
        self.summary = None
        self.original_path = None
        self.restored_path = None
        self.colorized_path = None
        self.pending_colorized_path = None
        self.pending_color_base_path = None
        self.pending_color_version_dir = None
        self.colorization_version_index = 0
        self.enhancement_base_path = None
        self.enhanced_path = None
        self.comparison_path = None
        self.enhancement_version = 0

        self.set_status("Photo loaded. Start restoration when ready.")
        self.set_step("upload", "Ready")
        self.restore_button.configure(state="normal")
        self.download_button.configure(state="disabled")
        self.upload_path_label.configure(text=str(self.uploaded_path))
        self.show_image(self.upload_preview, self.uploaded_path, "upload", 860, 560)
        self.notebook.select(0)

    def start_restoration(self) -> None:
        if not self.uploaded_path:
            messagebox.showerror("Missing photo", "Upload a photo first.")
            return

        backend = self.backend_var.get()
        run_config = apply_cli_overrides(load_config(), backend=backend)
        run_config["restoration"]["gpu"] = self.gpu_var.get().strip() or "-1"
        run_config["preprocess"]["model_safe_resize_longest_side"] = self.resolve_max_side()

        if backend == "boptl":
            ready, missing = backend_readiness(run_config["restoration"])
            if not ready:
                messagebox.showerror(
                    "Model assets missing",
                    "The Microsoft restoration model is not fully set up.\n\n"
                    + "\n".join(missing[:8]),
                )
                return

        self.restore_button.configure(state="disabled")
        self.upload_button.configure(state="disabled")
        self.set_status("Running restoration...")
        self.reset_process_steps()
        self.set_step("upload", "Running")

        worker = threading.Thread(
            target=self.restore_worker,
            args=(self.uploaded_path, run_config),
            daemon=True,
        )
        worker.start()

    def resolve_max_side(self) -> int:
        try:
            value = int(self.max_side_var.get())
        except (tk.TclError, ValueError):
            value = 768
        value = max(384, min(1600, value))
        self.max_side_var.set(value)
        return value

    def restore_worker(self, input_path: Path, run_config: dict) -> None:
        def progress(stage: str, message: str) -> None:
            self.event_queue.put(("progress", stage, message))

        try:
            summary = run_pipeline(
                str(input_path),
                config=build_restoration_only_config(run_config),
                progress_callback=progress,
            )
        except Exception as error:
            self.event_queue.put(("error", str(error)))
            return
        self.event_queue.put(("done", summary))

    def drain_events(self) -> None:
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break

            kind = event[0]
            if kind == "progress":
                _, stage, message = event
                self.handle_progress(stage, message)
            elif kind == "done":
                _, summary = event
                self.handle_restoration_done(summary)
            elif kind == "error":
                _, message = event
                self.handle_restoration_error(message)
            elif kind == "color_done":
                _, output_path, base_path, version_dir, message = event
                self.handle_colorization_done(
                    Path(output_path),
                    Path(base_path),
                    Path(version_dir),
                    message,
                )
            elif kind == "color_error":
                _, message = event
                self.handle_colorization_error(message)

        self.root.after(150, self.drain_events)

    def handle_progress(self, stage: str, message: str) -> None:
        if stage == "decision":
            stage = "restoration"
        if stage == "postprocess":
            stage = "restoration"
        if stage == "evaluation":
            stage = "comparison"
        if stage in self.step_labels:
            self.set_step(stage, "Running")
        self.set_status(message)

    def handle_restoration_done(self, summary: dict) -> None:
        self.summary = summary
        self.original_path = summary["input_validation"].copied_path
        self.restored_path = summary["restoration"].output_path
        self.colorized_path = None
        self.pending_colorized_path = None
        self.pending_color_base_path = None
        self.pending_color_version_dir = None
        self.colorization_version_index = 0
        self.enhancement_base_path = self.restored_path
        self.enhanced_path = summary["postprocess"].output_path

        self.set_step("upload", "Done")
        self.set_step("analysis", "Done")
        self.set_step("preprocess", "Done")
        self.set_step("restoration", "Done")
        self.set_step("comparison", "Ready")
        self.set_status("Restoration finished.")
        self.upload_button.configure(state="normal")
        self.restore_button.configure(state="normal")
        self.to_colorization_button.configure(state="normal")
        self.colorize_button.configure(state="normal")
        self.skip_color_button.configure(state="normal")
        self.keep_original_button.configure(state="normal")
        self.to_enhancement_button.configure(state="disabled")

        self.render_restoration_result()
        self.move_to_colorization()
        wants_color = messagebox.askyesno(
            "Colorization",
            "Restoration is finished. Do you want to colorize this photo?",
        )
        if not wants_color:
            self.skip_colorization()

    def handle_restoration_error(self, message: str) -> None:
        self.set_status("Restoration failed.")
        self.upload_button.configure(state="normal")
        self.restore_button.configure(state="normal")
        messagebox.showerror("Restoration failed", message)

    def render_restoration_result(self) -> None:
        if not self.summary:
            return

        validation = self.summary["input_validation"]
        analysis = self.summary["analysis"]
        preprocess = self.summary["preprocess"]
        decision = self.summary["decision"]
        restoration = self.summary["restoration"]

        self.show_image(self.original_preview, validation.copied_path, "original", 320, 360)
        self.show_image(self.preprocess_preview, preprocess.output_path, "preprocess", 320, 360)
        self.show_image(self.restored_preview, restoration.output_path, "restored", 320, 360)

        lines = []
        lines.extend(describe_validation(validation))
        lines.append("")
        lines.extend(describe_analysis(analysis))
        lines.append("")
        lines.extend(describe_preprocess(preprocess))
        lines.append("")
        lines.extend(describe_decision(decision))
        lines.append("")
        lines.extend(describe_restoration(restoration))
        self.write_text(self.restoration_text, "\n".join(lines))
        self.notebook.select(1)

    def move_to_colorization(self) -> None:
        if not self.restored_path:
            return
        self.set_step("colorization", "Ready")
        self.show_image(self.color_before_preview, self.restored_path, "color_before", 360, 440)
        preview_path = self.colorized_path or self.restored_path
        self.show_image(self.color_after_preview, preview_path, "color_after", 360, 440)
        self.select_recommended_palette(self.restored_path)
        self.notebook.select(2)

    def select_recommended_palette(self, image_path: Path) -> None:
        key = recommend_palette_key(load_image(image_path))
        for label, label_key in self.palette_labels.items():
            if label_key == key:
                self.palette_var.set(label)
                break
        self.update_palette_note()

    def apply_colorization(self) -> None:
        if not self.summary or not self.restored_path:
            return

        ready, missing = colorization_assets_ready(self.config["postprocess"]["colorization"])
        if not ready:
            messagebox.showerror(
                "DeOldify is not ready",
                "The DeOldify model cannot run until these assets/dependencies are available:\n\n"
                + "\n".join(missing[:8]),
            )
            self.set_status("DeOldify is not ready. Keep the restored original or set up the model.")
            return

        run_root = Path(self.summary["run_root"])
        restored_path = self.restored_path
        self.colorization_version_index += 1
        version_index = self.colorization_version_index

        self.set_step("colorization", "Running")
        self.set_status(
            "Running DeOldify model..."
        )
        self.colorize_button.configure(state="disabled")
        self.update_post_palette_button.configure(state="disabled")
        self.keep_color_button.configure(state="disabled")
        self.keep_original_button.configure(state="disabled")
        self.skip_color_button.configure(state="disabled")
        self.to_enhancement_button.configure(state="disabled")

        worker = threading.Thread(
            target=self.colorization_worker,
            args=(
                restored_path,
                run_root,
                version_index,
            ),
            daemon=True,
        )
        worker.start()

    def colorization_worker(
        self,
        restored_path: Path,
        run_root: Path,
        version_index: int,
    ) -> None:
        try:
            before_palette_key = None
            before_intensity = None
            after_palette_key = None
            after_intensity = None
            use_deoldify = True
            stage_dir = run_root / "06_postprocess"
            versions_dir = stage_dir / "colorization_versions"
            version_dir = versions_dir / f"v{version_index:02d}_model"
            version_dir.mkdir(parents=True, exist_ok=True)
            input_image = load_image(restored_path)
            colorization_config = copy.deepcopy(self.config["postprocess"]["colorization"])

            result = apply_staged_colorization(
                image=input_image,
                colorization_config=colorization_config,
                stage_dir=stage_dir,
                use_deoldify=use_deoldify,
                before_palette_key=before_palette_key,
                before_intensity=0.0,
                after_palette_key=after_palette_key,
                after_intensity=0.0,
            )
            base_path = save_image(
                stage_dir / "colorization_base_preview.png",
                result.base_image,
            ).resolve()
            output_path = save_image(
                stage_dir / "colorized_preview.png",
                result.output_image,
            ).resolve()

            save_json(
                stage_dir / "desktop_colorization_preview.json",
                {
                    "output_path": output_path,
                    "base_path": base_path,
                    "before_palette_key": before_palette_key,
                    "before_intensity": before_intensity,
                    "after_palette_key": after_palette_key,
                    "after_intensity": after_intensity,
                    "requested_deoldify": use_deoldify,
                    "used_deoldify": result.used_deoldify,
                    "notes": result.notes,
                },
            )
            save_image(version_dir / "restored_input.png", input_image)
            save_image(version_dir / "model_input.png", result.pre_model_image)
            version_base_path = save_image(version_dir / "deoldify_base.png", result.base_image).resolve()
            version_output_path = save_image(version_dir / "preview_output.png", result.output_image).resolve()
            save_json(
                version_dir / "metadata.json",
                {
                    "version": version_index,
                    "output_path": version_output_path,
                    "base_path": version_base_path,
                    "model_input_path": (version_dir / "model_input.png").resolve(),
                    "restored_input_path": (version_dir / "restored_input.png").resolve(),
                    "before_palette_key": before_palette_key,
                    "before_intensity": before_intensity,
                    "after_palette_key": after_palette_key,
                    "after_intensity": after_intensity,
                    "requested_deoldify": use_deoldify,
                    "used_deoldify": result.used_deoldify,
                    "notes": result.notes,
                },
            )
            self.rebuild_colorization_version_sheets(versions_dir)

            if result.used_deoldify:
                message = "DeOldify model output is ready. Add a palette or keep this version."
            else:
                message = "DeOldify did not produce a model output. Keep the restored original or retry."
            self.event_queue.put(("color_done", output_path, base_path, version_dir, message))
        except Exception as error:
            self.event_queue.put(("color_error", str(error)))

    def handle_colorization_done(
        self,
        output_path: Path,
        base_path: Path,
        version_dir: Path,
        message: str,
    ) -> None:
        version_output_path = version_dir / "preview_output.png"
        version_base_path = version_dir / "deoldify_base.png"
        self.pending_colorized_path = version_output_path if version_output_path.exists() else output_path
        self.pending_color_base_path = version_base_path if version_base_path.exists() else base_path
        self.pending_color_version_dir = version_dir
        self.set_step("colorization", "Preview")
        self.set_status(message)
        self.show_image(self.color_after_preview, self.pending_colorized_path, "color_after", 360, 440)
        self.colorize_button.configure(state="normal")
        self.update_post_palette_button.configure(state="normal")
        self.keep_color_button.configure(state="normal")
        self.keep_original_button.configure(state="normal")
        self.skip_color_button.configure(state="normal")
        self.to_enhancement_button.configure(state="disabled")

    def handle_colorization_error(self, message: str) -> None:
        self.set_step("colorization", "Ready")
        self.set_status("Colorization failed.")
        self.colorize_button.configure(state="normal")
        self.update_post_palette_button.configure(state="normal" if self.pending_color_base_path else "disabled")
        self.keep_color_button.configure(state="normal" if self.pending_colorized_path else "disabled")
        self.keep_original_button.configure(state="normal")
        self.skip_color_button.configure(state="normal")
        self.to_enhancement_button.configure(state="disabled")
        messagebox.showerror("Colorization failed", message)

    def read_colorization_metadata(self, version_dir: Path | None) -> dict:
        if not version_dir:
            return {}
        metadata_path = version_dir / "metadata.json"
        if not metadata_path.exists():
            return {}
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def rebuild_colorization_version_sheets(self, versions_dir: Path) -> None:
        try:
            final_entries = []
            model_entries = []
            for metadata_path in sorted(versions_dir.glob("v*/metadata.json")):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                version = int(metadata.get("version", 0))
                after = metadata.get("after_palette_key") or "none"
                label = f"v{version:02d} after:{after[:10]}"

                output_path = Path(metadata["output_path"])
                if output_path.exists():
                    final_entries.append((label, load_image(output_path)))

                base_path = Path(metadata["base_path"])
                if base_path.exists():
                    model_entries.append((f"v{version:02d} model", load_image(base_path)))

            if final_entries:
                final_grid = create_comparison_grid(final_entries)
                save_image(versions_dir / "colorization_versions_comparison.png", final_grid)
            if model_entries:
                model_grid = create_comparison_grid(model_entries)
                save_image(versions_dir / "deoldify_base_comparison.png", model_grid)
        except Exception:
            return

    def update_after_palette_preview(self) -> None:
        if not self.summary or not self.pending_color_base_path:
            messagebox.showinfo(
                "No preview yet",
                "Run the DeOldify model first, then add an after-model palette.",
            )
            return

        stage_dir = Path(self.summary["run_root"]) / "06_postprocess"
        versions_dir = stage_dir / "colorization_versions"
        base_image = load_image(self.pending_color_base_path)
        palette_key = self.palette_labels[self.palette_var.get()]
        result = apply_palette_colorization(
            base_image,
            palette_key=palette_key,
            intensity=float(self.intensity_var.get()),
        )
        output_image = result.output_image
        notes = result.notes

        output_path = save_image(stage_dir / "colorized_preview.png", output_image).resolve()
        self.colorization_version_index += 1
        version_index = self.colorization_version_index
        previous_metadata = self.read_colorization_metadata(self.pending_color_version_dir)
        version_dir = versions_dir / f"v{version_index:02d}_after-{slug_part(palette_key)}"
        version_dir.mkdir(parents=True, exist_ok=True)

        restored_input_path = previous_metadata.get("restored_input_path")
        model_input_path = previous_metadata.get("model_input_path")
        if restored_input_path and Path(restored_input_path).exists():
            save_image(version_dir / "restored_input.png", load_image(Path(restored_input_path)))
        if model_input_path and Path(model_input_path).exists():
            save_image(version_dir / "model_input.png", load_image(Path(model_input_path)))
        version_base_path = save_image(version_dir / "deoldify_base.png", base_image).resolve()
        version_output_path = save_image(version_dir / "preview_output.png", output_image).resolve()

        save_json(
            stage_dir / "desktop_after_palette_preview.json",
            {
                "output_path": output_path,
                "base_path": self.pending_color_base_path,
                "after_palette_key": palette_key,
                "after_intensity": float(self.intensity_var.get()),
                "notes": notes,
            },
        )
        save_json(
            version_dir / "metadata.json",
            {
                "version": version_index,
                "output_path": version_output_path,
                "base_path": version_base_path,
                "model_input_path": (
                    (version_dir / "model_input.png").resolve()
                    if (version_dir / "model_input.png").exists()
                    else None
                ),
                "restored_input_path": (
                    (version_dir / "restored_input.png").resolve()
                    if (version_dir / "restored_input.png").exists()
                    else None
                ),
                "before_palette_key": None,
                "before_intensity": None,
                "after_palette_key": palette_key,
                "after_intensity": float(self.intensity_var.get()),
                "requested_deoldify": previous_metadata.get("requested_deoldify", False),
                "used_deoldify": previous_metadata.get("used_deoldify", False),
                "notes": notes,
            },
        )
        self.rebuild_colorization_version_sheets(versions_dir)

        self.pending_colorized_path = version_output_path
        self.pending_color_version_dir = version_dir
        self.show_image(self.color_after_preview, self.pending_colorized_path, "color_after", 360, 440)
        self.keep_color_button.configure(state="normal")
        self.keep_original_button.configure(state="normal")
        self.set_status("After-palette preview updated. Keep it or keep the restored original.")

    def keep_colorization_preview(self) -> None:
        if not self.summary or not self.pending_colorized_path:
            messagebox.showinfo("No preview yet", "Run a colorization preview first.")
            return

        stage_dir = Path(self.summary["run_root"]) / "06_postprocess"
        self.colorized_path = self.pending_colorized_path
        save_image(stage_dir / "colorized_output.png", load_image(self.colorized_path))
        self.set_step("colorization", "Done")
        self.set_status("Colorization preview kept. Enhancement is ready.")
        self.show_image(self.color_after_preview, self.colorized_path, "color_after", 360, 440)
        self.to_enhancement_button.configure(state="normal")
        self.download_button.configure(state="disabled")
        self.move_to_enhancement()

    def keep_restored_original(self) -> None:
        if not self.restored_path:
            return
        self.colorized_path = self.restored_path
        self.pending_colorized_path = None
        self.pending_color_base_path = None
        self.set_step("colorization", "Original")
        self.set_status("Restored original kept. Enhancement is ready.")
        self.show_image(self.color_after_preview, self.colorized_path, "color_after", 360, 440)
        self.keep_color_button.configure(state="disabled")
        self.update_post_palette_button.configure(state="disabled")
        self.to_enhancement_button.configure(state="normal")
        self.move_to_enhancement()

    def skip_colorization(self) -> None:
        if not self.restored_path:
            return
        self.keep_restored_original()

    def move_to_enhancement(self) -> None:
        if not self.restored_path:
            return

        self.enhancement_base_path = self.colorized_path or self.restored_path
        self.enhanced_path = self.enhancement_base_path
        self.set_step("enhancement", "Ready")
        self.reset_enhancement_defaults()
        self.show_image(self.enhance_before_preview, self.enhancement_base_path, "enhance_before", 380, 500)
        self.show_image(self.enhance_after_preview, self.enhanced_path, "enhance_after", 380, 500)
        self.enhance_button.configure(state="normal")
        self.enhance_again_button.configure(state="normal")
        self.to_final_button.configure(state="normal")
        self.notebook.select(3)

    def reset_enhancement_defaults(self) -> None:
        if not self.enhancement_base_path:
            return

        image = load_image(self.enhancement_base_path)
        settings = build_recommended_enhancement_settings(image)
        for key, value in settings.items():
            if key in self.slider_vars:
                self.slider_vars[key].set(float(value))
                self.update_slider_label(key)

        palette_key = recommend_palette_key(image)
        palette = PALETTE_PRESETS[palette_key]
        summary = describe_enhancement_recommendation(image)
        self.recommended_palette_var.set(
            f"{summary} Suggested palette: {palette['name']}."
        )

    def apply_enhancement(self, use_current_output: bool) -> None:
        if not self.summary or not self.enhancement_base_path:
            return

        source_path = self.enhanced_path if use_current_output and self.enhanced_path else self.enhancement_base_path
        if not source_path:
            return

        self.enhancement_version += 1
        settings = self.current_enhancement_settings()
        stage_dir = Path(self.summary["run_root"]) / "06_postprocess"
        output_name = f"enhanced_{self.enhancement_version:02d}.png"
        enhanced = apply_enhancement_controls(load_image(source_path), settings)
        output_path = save_image(stage_dir / output_name, enhanced).resolve()
        final_path = save_image(stage_dir / "final_restored.png", enhanced).resolve()
        save_json(
            stage_dir / "enhancement_controls.json",
            {
                "source_path": source_path,
                "output_path": output_path,
                "final_path": final_path,
                "settings": settings,
            },
        )

        self.enhanced_path = final_path
        self.set_step("enhancement", "Done")
        self.set_status("Enhancement updated.")
        self.show_image(self.enhance_after_preview, self.enhanced_path, "enhance_after", 380, 500)
        self.update_comparison()
        self.download_button.configure(state="normal")

    def move_to_final_view(self) -> None:
        self.update_comparison()
        self.download_button.configure(state="normal")
        self.set_step("comparison", "Done")
        self.notebook.select(4)

    def update_comparison(self) -> None:
        if not self.summary or not self.original_path or not self.restored_path:
            return

        color_path = self.colorized_path or self.restored_path
        final_path = self.enhanced_path or color_path

        self.show_image(self.compare_original_preview, self.original_path, "compare_original", 245, 300)
        self.show_image(self.compare_restored_preview, self.restored_path, "compare_restored", 245, 300)
        self.show_image(self.compare_color_preview, color_path, "compare_color", 245, 300)
        self.show_image(self.compare_final_preview, final_path, "compare_final", 245, 300)

        comparison = create_comparison_grid(
            [
                ("Original", load_image(self.original_path)),
                ("After Restoration", load_image(self.restored_path)),
                ("After Colorization", load_image(color_path)),
                ("After Final Touches", load_image(final_path)),
            ]
        )
        evaluation_dir = Path(self.summary["run_root"]) / "07_evaluation"
        self.comparison_path = save_image(evaluation_dir / "desktop_stage_comparison.png", comparison).resolve()
        self.show_image(self.comparison_full_preview, self.comparison_path, "comparison_full", 820, 360)

    def export_final_product(self) -> None:
        final_path = self.enhanced_path or self.colorized_path or self.restored_path
        if not final_path:
            messagebox.showerror("No final image", "There is no finished image to export yet.")
            return

        destination = filedialog.asksaveasfilename(
            title="Save final image",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("All files", "*.*")],
            initialfile="photo_reviver_final.png",
        )
        if not destination:
            return
        shutil.copy2(final_path, destination)
        self.set_status(f"Final product saved to {destination}")

    def current_enhancement_settings(self) -> dict[str, int]:
        return {
            key: int(round(var.get()))
            for key, var in self.slider_vars.items()
        }

    def update_palette_note(self, _event: object | None = None) -> None:
        selected = self.palette_var.get()
        key = self.palette_labels.get(selected)
        if not key:
            return
        preset = PALETTE_PRESETS[key]
        self.palette_note_var.set(preset["description"])

    def update_intensity_label(self) -> None:
        self.intensity_label.configure(text=f"{self.intensity_var.get():.2f}")

    def update_deoldify_note(self) -> None:
        ready, missing = colorization_assets_ready(self.config["postprocess"]["colorization"])
        if ready:
            self.deoldify_note_var.set(
                "DeOldify assets detected. Run Model creates a clean model preview before any palette is added."
            )
            return

        preview = ", ".join(missing[:3])
        if len(missing) > 3:
            preview = f"{preview}, ..."
        self.deoldify_note_var.set(
            "DeOldify is not ready yet, so Run Model is unavailable until setup is complete. "
            f"Missing: {preview}"
        )

    def update_slider_label(self, key: str) -> None:
        if key in self.slider_value_labels:
            self.slider_value_labels[key].configure(text=str(int(round(self.slider_vars[key].get()))))

    def show_image(
        self,
        label: ttk.Label,
        path: Path,
        ref_key: str,
        max_width: int,
        max_height: int,
    ) -> None:
        photo = image_to_photo(path, max_width, max_height)
        self.photo_refs[ref_key] = photo
        label.configure(image=photo)

    def write_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def reset_process_steps(self) -> None:
        for key, label in PROCESS_STEPS:
            self.set_step(key, "Waiting")

    def set_step(self, key: str, status: str) -> None:
        label = self.step_labels.get(key)
        if label:
            display = next((name for item_key, name in PROCESS_STEPS if item_key == key), key.title())
            label.configure(text=f"{status} - {display}")


def main() -> None:
    root = tk.Tk()
    PhotoReviverDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
