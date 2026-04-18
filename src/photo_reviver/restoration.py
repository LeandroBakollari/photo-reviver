from __future__ import annotations

import subprocess
from pathlib import Path

from photo_reviver.io_utils import load_image, save_image, save_json
from photo_reviver.types import RestorationResult


class PassthroughRestorationRunner:
    backend = "passthrough"

    def run(self, input_path: Path, mode: str, stage_dir: Path) -> RestorationResult:
        image = load_image(input_path)
        output_path = save_image(stage_dir / "restored_model_output.png", image)

        result = RestorationResult(
            output_path=output_path.resolve(),
            backend=self.backend,
            notes=[
                "No model was used.",
                f"The restoration stage is a pass-through placeholder for mode '{mode}'.",
            ],
        )
        save_json(stage_dir / "restoration.json", result)
        return result


class ExternalCommandRestorationRunner:
    backend = "external_command"

    def __init__(self, command_template: list[str]) -> None:
        self.command_template = command_template

    def run(self, input_path: Path, mode: str, stage_dir: Path) -> RestorationResult:
        output_path = (stage_dir / "restored_model_output.png").resolve()
        log_path = (stage_dir / "external_command.log").resolve()

        command = [
            part.format(
                input_path=str(input_path.resolve()),
                output_path=str(output_path),
                mode=mode,
                stage_dir=str(stage_dir.resolve()),
            )
            for part in self.command_template
        ]

        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        log_text = completed.stdout
        if completed.stderr:
            log_text = f"{log_text}\n\n[stderr]\n{completed.stderr}".strip()
        log_path.write_text(log_text, encoding="utf-8")

        if not output_path.exists():
            raise FileNotFoundError(
                "External restoration command finished, but the expected output file "
                f"was not created: {output_path}"
            )

        result = RestorationResult(
            output_path=output_path,
            backend=self.backend,
            notes=["Restoration was delegated to an external command template."],
            command=command,
            log_path=log_path,
        )
        save_json(stage_dir / "restoration.json", result)
        return result


def build_restoration_runner(restoration_config: dict):
    backend = restoration_config.get("backend", "passthrough")
    if backend == "passthrough":
        return PassthroughRestorationRunner()
    if backend == "external_command":
        command_template = restoration_config.get("external_command")
        if not command_template:
            raise ValueError("Missing 'external_command' template in config.")
        return ExternalCommandRestorationRunner(command_template)

    raise ValueError(f"Unsupported restoration backend: {backend}")
