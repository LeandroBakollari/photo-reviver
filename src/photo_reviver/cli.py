from __future__ import annotations

import argparse

from photo_reviver.config import apply_cli_overrides, load_config
from photo_reviver.io_utils import json_ready
from photo_reviver.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Photo Reviver scaffold pipeline on one image."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Full path to the old photo you want to process.",
    )
    parser.add_argument(
        "--reference",
        help="Optional clean reference image for metric calculation.",
    )
    parser.add_argument(
        "--config",
        help="Optional path to a JSON config file.",
    )
    parser.add_argument(
        "--output-root",
        help="Optional output root folder. Default: artifacts/runs",
    )
    parser.add_argument(
        "--backend",
        choices=["passthrough", "external_command"],
        help="Override the restoration backend from the config file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_cli_overrides(
        config,
        output_root=args.output_root,
        backend=args.backend,
    )

    result = run_pipeline(
        input_path=args.input,
        config=config,
        reference_path=args.reference,
    )
    summary = json_ready(result)

    print("Photo Reviver run completed.")
    print(f"Run folder: {summary['run_root']}")
    print(f"Chosen mode: {summary['decision']['mode']}")
    print(f"Final image: {summary['postprocess']['output_path']}")
    print(f"Comparison image: {summary['evaluation']['comparison_path']}")


if __name__ == "__main__":
    main()
