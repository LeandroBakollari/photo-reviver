from __future__ import annotations

from pathlib import Path

from photo_reviver.analysis import analyze_image
from photo_reviver.decision import choose_restoration_mode
from photo_reviver.evaluate import evaluate_result
from photo_reviver.io_utils import (
    build_run_paths,
    copy_input_file,
    load_image,
    save_json,
    validate_image,
)
from photo_reviver.postprocess import postprocess_image
from photo_reviver.preprocess import preprocess_image
from photo_reviver.restoration import build_restoration_runner


def run_pipeline(
    input_path: str,
    config: dict,
    reference_path: str | None = None,
) -> dict:
    source_path = Path(input_path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Input image not found: {source_path}")

    output_root = Path(config["paths"]["output_root"])
    run_paths = build_run_paths(output_root, source_path.stem)

    copied_input = copy_input_file(source_path, run_paths.input_dir)
    image = load_image(copied_input)
    validation = validate_image(
        source_path=source_path,
        copied_path=copied_input,
        image=image,
        min_width=int(config["analysis"]["min_width"]),
        min_height=int(config["analysis"]["min_height"]),
    )

    analysis = analyze_image(
        image=image,
        validation=validation,
        analysis_config=config["analysis"],
        stage_dir=run_paths.analysis_dir,
    )

    preprocess_result = preprocess_image(
        image=image,
        analysis=analysis,
        backend=config["restoration"]["backend"],
        preprocess_config=config["preprocess"],
        stage_dir=run_paths.preprocess_dir,
    )

    decision = choose_restoration_mode(analysis)
    save_json(run_paths.decision_dir / "decision.json", decision)

    restoration_runner = build_restoration_runner(config["restoration"])
    restoration_result = restoration_runner.run(
        input_path=preprocess_result.output_path,
        mode=decision.mode,
        stage_dir=run_paths.restoration_dir,
    )

    restored_image = load_image(restoration_result.output_path)
    postprocess_result = postprocess_image(
        image=restored_image,
        postprocess_config=config["postprocess"],
        stage_dir=run_paths.postprocess_dir,
    )

    final_image = load_image(postprocess_result.output_path)
    evaluation_result = evaluate_result(
        original_image=image,
        restored_image=restored_image,
        final_image=final_image,
        stage_dir=run_paths.evaluation_dir,
        reference_path=Path(reference_path).expanduser() if reference_path else None,
    )

    summary = {
        "run_root": run_paths.run_root.resolve(),
        "input_validation": validation,
        "analysis": analysis,
        "decision": decision,
        "preprocess": preprocess_result,
        "restoration": restoration_result,
        "postprocess": postprocess_result,
        "evaluation": evaluation_result,
    }
    save_json(run_paths.run_root / "run_summary.json", summary)
    return summary


def rerun_final_touches(summary: dict, postprocess_config: dict) -> dict:
    run_root = Path(summary["run_root"])
    postprocess_dir = run_root / "06_postprocess"
    evaluation_dir = run_root / "07_evaluation"

    restored_image = load_image(summary["restoration"].output_path)
    postprocess_result = postprocess_image(
        image=restored_image,
        postprocess_config=postprocess_config,
        stage_dir=postprocess_dir,
    )

    original_image = load_image(summary["input_validation"].copied_path)
    restored_image = load_image(summary["restoration"].output_path)
    final_image = load_image(postprocess_result.output_path)
    reference_path = summary["evaluation"].metrics.reference_path

    evaluation_result = evaluate_result(
        original_image=original_image,
        restored_image=restored_image,
        final_image=final_image,
        stage_dir=evaluation_dir,
        reference_path=reference_path,
    )

    summary["postprocess"] = postprocess_result
    summary["evaluation"] = evaluation_result
    save_json(run_root / "run_summary.json", summary)
    return summary
