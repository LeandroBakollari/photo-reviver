from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import streamlit as st

from photo_reviver.app_utils import (
    backend_readiness,
    colorization_readiness,
    describe_analysis,
    describe_decision,
    describe_postprocess,
    describe_preprocess,
    describe_restoration,
    describe_validation,
)
from photo_reviver.config import apply_cli_overrides, load_config
from photo_reviver.io_utils import load_image
from photo_reviver.pipeline import rerun_final_touches, run_pipeline


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --paper: #f5efe2;
          --ink: #2f251d;
          --muted: #6f6154;
          --accent: #8f5b3c;
          --line: #d4c2ab;
        }
        .stApp {
          background:
            radial-gradient(circle at top left, rgba(190, 160, 120, 0.18), transparent 30%),
            linear-gradient(180deg, #f8f2e8 0%, #f2e6d6 100%);
        }
        .block-container {
          max-width: 1180px;
          padding-top: 2rem;
          padding-bottom: 2rem;
        }
        h1, h2, h3 {
          font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
          color: var(--ink);
        }
        p, li, label, div {
          color: var(--ink);
        }
        .hero {
          padding: 1.5rem 1.6rem;
          border: 1px solid var(--line);
          border-radius: 24px;
          background:
            linear-gradient(135deg, rgba(255,255,255,0.7), rgba(232,214,188,0.92));
          box-shadow: 0 14px 35px rgba(80, 55, 35, 0.08);
          margin-bottom: 1rem;
        }
        .hero p {
          color: var(--muted);
          font-size: 1rem;
          margin-bottom: 0;
        }
        .stage-shell {
          background: rgba(255, 252, 247, 0.82);
          border: 1px solid var(--line);
          border-radius: 22px;
          padding: 1rem 1.1rem;
          box-shadow: 0 10px 24px rgba(60, 45, 25, 0.05);
        }
        .tiny-note {
          color: var(--muted);
          font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def resolve_backend_choice() -> str:
    config = load_config()
    boptl_ready, _ = backend_readiness(config["restoration"])
    options = ["boptl", "passthrough"] if boptl_ready else ["passthrough", "boptl"]
    labels = {
        "boptl": "Microsoft model (best quality, slower)",
        "passthrough": "Simple pipeline only (fast, no learned restoration)",
    }
    selected_label = st.sidebar.radio(
        "Restoration engine",
        [labels[item] for item in options],
        index=0,
    )
    return next(key for key, label in labels.items() if label == selected_label)


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def render_lines(lines: list[str]) -> None:
    for line in lines:
        st.markdown(f"- {line}")


def render_stage_box(title: str, subtitle: str | None = None) -> None:
    st.markdown('<div class="stage-shell">', unsafe_allow_html=True)
    st.subheader(title)
    if subtitle:
        st.markdown(f'<p class="tiny-note">{subtitle}</p>', unsafe_allow_html=True)


def close_stage_box() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_image(image_path: Path, caption: str, use_container_width: bool = True) -> None:
    image = load_image(image_path)
    if image.ndim == 3:
        image = image[:, :, ::-1]
    st.image(image, caption=caption, use_container_width=use_container_width)


def unpack_summary(summary: dict):
    validation = summary["input_validation"]
    analysis = summary["analysis"]
    decision = summary["decision"]
    preprocess = summary["preprocess"]
    restoration = summary["restoration"]
    postprocess = summary["postprocess"]
    evaluation = summary["evaluation"]

    original_path = validation.copied_path
    preprocessed_path = preprocess.output_path
    restored_path = restoration.output_path
    final_path = postprocess.output_path

    return (
        validation,
        analysis,
        decision,
        preprocess,
        restoration,
        postprocess,
        evaluation,
        original_path,
        preprocessed_path,
        restored_path,
        final_path,
    )


def build_clean_run_config(config: dict) -> dict:
    run_config = copy.deepcopy(config)
    run_config["postprocess"]["apply_enhancement"] = False
    run_config["postprocess"]["enhancement_strength"] = 0.0
    run_config["postprocess"]["apply_sharpening"] = False
    run_config["postprocess"]["sharpening_strength"] = 0.0
    run_config["postprocess"]["simple_upscale_factor"] = 1.0
    run_config["postprocess"]["attempt_colorization"] = False
    run_config["postprocess"]["colorization"]["enabled"] = False
    return run_config


def build_final_touches_config(
    config: dict,
    enhancement_strength: float,
    sharpening_strength: float,
    apply_colorization: bool,
) -> dict:
    postprocess_config = copy.deepcopy(config["postprocess"])
    postprocess_config["apply_enhancement"] = enhancement_strength > 0.0
    postprocess_config["enhancement_strength"] = enhancement_strength
    postprocess_config["apply_sharpening"] = sharpening_strength > 0.0
    postprocess_config["sharpening_strength"] = sharpening_strength
    postprocess_config["attempt_colorization"] = apply_colorization
    postprocess_config["colorization"]["enabled"] = apply_colorization
    return postprocess_config


def render_result(summary: dict, config: dict) -> None:
    (
        validation,
        analysis,
        decision,
        preprocess,
        restoration,
        postprocess,
        evaluation,
        original_path,
        preprocessed_path,
        restored_path,
        final_path,
    ) = unpack_summary(summary)

    run_root = Path(summary["run_root"])
    run_key = run_root.name

    st.success("Fix completed. You can review each stage below.")
    st.caption(f"Run folder: {summary['run_root']}")

    tabs = st.tabs(
        [
            "1. Uploaded",
            "2. Analysis",
            "3. Preprocess",
            "4. Mode",
            "5. Restoration",
            "6. Final Touches",
            "7. Final Result",
        ]
    )

    with tabs[0]:
        render_stage_box("Uploaded Photo", "The app validates the image and stores a clean copy for the run.")
        render_image(original_path, "Uploaded Photo")
        render_lines(describe_validation(validation))
        close_stage_box()

    with tabs[1]:
        render_stage_box("Image Analysis", "These views help explain why the pipeline chose a certain restoration path.")
        gray_col, hist_col, scratch_col, overlay_col = st.columns(4)
        with gray_col:
            render_image(analysis.grayscale_path, "Grayscale View")
        with hist_col:
            render_image(analysis.histogram_path, "Histogram")
        with scratch_col:
            render_image(analysis.scratch_mask_path, "Scratch Mask")
        with overlay_col:
            render_image(analysis.scratch_overlay_path, "Scratch Overlay")
        render_lines(describe_analysis(analysis))
        close_stage_box()

    with tabs[2]:
        render_stage_box("Preprocess", "This stage gently prepares the image before restoration.")
        before_col, after_col = st.columns(2)
        with before_col:
            render_image(original_path, "Original")
        with after_col:
            render_image(preprocessed_path, "Preprocessed")
        render_lines(describe_preprocess(preprocess))
        close_stage_box()

    with tabs[3]:
        render_stage_box("Chosen Mode", "The pipeline turns the analysis into a simple restoration decision.")
        st.metric("Selected Mode", decision.mode)
        render_lines(describe_decision(decision))
        close_stage_box()

    with tabs[4]:
        render_stage_box("Restoration Output", "This is the image right after the restoration engine runs.")
        before_col, after_col = st.columns(2)
        with before_col:
            render_image(preprocessed_path, "Input To Restoration")
        with after_col:
            render_image(restored_path, "Restoration Output")
        render_lines(describe_restoration(restoration))
        close_stage_box()

    with tabs[5]:
        render_stage_box("Final Touches", "These controls work on the image after the restoration model finishes.")
        controls_col, preview_col = st.columns([0.9, 1.1])

        with controls_col:
            enhancement_strength = st.slider(
                "Enhancement strength",
                min_value=0.0,
                max_value=0.35,
                value=float(config["postprocess"]["enhancement_strength"]),
                step=0.01,
                key=f"enhancement_{run_key}",
            )
            sharpening_strength = st.slider(
                "Sharpening strength",
                min_value=0.0,
                max_value=0.25,
                value=float(config["postprocess"]["sharpening_strength"]),
                step=0.01,
                key=f"sharpening_{run_key}",
            )

            color_ready, color_missing = colorization_readiness(config["postprocess"])
            if color_ready:
                st.caption("DeOldify colorization is ready.")
            else:
                st.caption("DeOldify colorization is not ready yet.")
                for item in color_missing:
                    st.caption(item)

            apply_button = st.button(
                "Apply Final Touches",
                use_container_width=True,
                key=f"apply_touches_{run_key}",
            )
            colorize_button = st.button(
                "Run DeOldify Colorization",
                use_container_width=True,
                key=f"apply_color_{run_key}",
                disabled=not color_ready,
            )

        if apply_button or colorize_button:
            action_label = (
                "Running DeOldify colorization..."
                if colorize_button
                else "Applying final touches..."
            )
            with st.spinner(action_label):
                postprocess_config = build_final_touches_config(
                    config,
                    enhancement_strength=enhancement_strength,
                    sharpening_strength=sharpening_strength,
                    apply_colorization=colorize_button,
                )
                summary = rerun_final_touches(summary, postprocess_config)
                st.session_state.latest_summary = summary
                (
                    validation,
                    analysis,
                    decision,
                    preprocess,
                    restoration,
                    postprocess,
                    evaluation,
                    original_path,
                    preprocessed_path,
                    restored_path,
                    final_path,
                ) = unpack_summary(summary)

            if colorize_button:
                st.success("DeOldify colorization finished.")
            else:
                st.success("Final touches updated.")

        with preview_col:
            before_col, after_col = st.columns(2)
            with before_col:
                render_image(restored_path, "After Restoration")
            with after_col:
                render_image(final_path, "After Final Touches")
            if postprocess.colorized_path:
                render_image(postprocess.colorized_path, "DeOldify Colorized Output")

        render_lines(describe_postprocess(postprocess))
        close_stage_box()

    with tabs[6]:
        render_stage_box("Final Result", "Here is the finished image and the main comparison views.")
        stage_col_1, stage_col_2, stage_col_3, stage_col_4 = st.columns(4)
        with stage_col_1:
            render_image(original_path, "Original")
        with stage_col_2:
            render_image(analysis.grayscale_path, "Grayscale")
        with stage_col_3:
            render_image(restored_path, "After Restoration Model")
        with stage_col_4:
            render_image(final_path, "After Final Touches")
        render_image(evaluation.comparison_path, "Stage-By-Stage Comparison")
        with open(final_path, "rb") as file:
            st.download_button(
                "Download Final Image",
                data=file.read(),
                file_name=final_path.name,
                mime="image/png",
            )
        close_stage_box()


def main() -> None:
    st.set_page_config(
        page_title="Photo Reviver",
        page_icon="photo",
        layout="wide",
    )
    inject_styles()

    st.markdown(
        """
        <div class="hero">
          <h1>Photo Reviver</h1>
          <p>Upload a damaged photo, press <strong>Fix Photo</strong>, then adjust the final touches after the restoration model finishes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "latest_summary" not in st.session_state:
        st.session_state.latest_summary = None

    backend = resolve_backend_choice()
    config = apply_cli_overrides(load_config(), backend=backend)

    if backend == "boptl":
        ready, missing = backend_readiness(config["restoration"])
        config["restoration"]["gpu"] = st.sidebar.text_input("GPU ids", value="-1")
        if ready:
            st.sidebar.success("Microsoft model assets found.")
        else:
            st.sidebar.error("Microsoft model is not fully set up.")
            for path in missing:
                st.sidebar.caption(path)

    uploaded_file = st.file_uploader(
        "Upload a damaged photo",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        help="The app will copy the uploaded file into a run folder and process it from there.",
    )

    if uploaded_file is not None:
        preview_col, note_col = st.columns([1.2, 0.8])
        preview_col.image(uploaded_file, caption="Uploaded Preview", use_container_width=True)
        note_col.markdown(
            """
            ### What Happens Next
            - the image is validated
            - the pipeline analyzes scratches, contrast, and faces
            - the restoration engine runs
            - final touches are adjusted after the model output
            """
        )

    disabled = uploaded_file is None or (backend == "boptl" and not backend_readiness(config["restoration"])[0])
    if st.button("Fix Photo", type="primary", use_container_width=True, disabled=disabled):
        temp_path = save_uploaded_file(uploaded_file)
        try:
            with st.status("Running the restoration pipeline...", expanded=True) as status:
                status.write("Saving the uploaded image into a temporary file.")
                status.write("Analyzing the photo and preparing the restoration run.")
                summary = run_pipeline(
                    str(temp_path),
                    config=build_clean_run_config(config),
                )
                status.write("The model output is ready for final-touch editing.")
                status.update(label="Restoration finished.", state="complete")
            st.session_state.latest_summary = summary
        except Exception as error:  # pragma: no cover - UI path
            st.error(f"The run failed: {error}")
        finally:
            temp_path.unlink(missing_ok=True)

    if st.session_state.latest_summary is not None:
        render_result(st.session_state.latest_summary, config)


if __name__ == "__main__":
    main()
