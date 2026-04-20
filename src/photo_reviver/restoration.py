from __future__ import annotations

import shutil
import subprocess
import tempfile
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


def build_boptl_command(
    python_executable: str,
    checkpoint_name: str,
    input_dir: Path,
    output_dir: Path,
    gpu: str,
    mode: str,
) -> list[str]:
    command = [
        python_executable,
        "run.py",
        "--input_folder",
        str(input_dir),
        "--output_folder",
        str(output_dir),
        "--GPU",
        gpu,
        "--checkpoint_name",
        checkpoint_name,
    ]

    if mode in {"scratch", "scratch+hr"}:
        command.append("--with_scratch")
    if mode == "scratch+hr":
        command.append("--HR")

    return command


class BoptlRestorationRunner:
    backend = "boptl"

    def __init__(
        self,
        repo_root: str,
        python_executable: str = "python",
        gpu: str = "-1",
        checkpoint_name: str = "Setting_9_epoch_100",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.python_executable = python_executable
        self.gpu = gpu
        self.checkpoint_name = checkpoint_name

    def run(self, input_path: Path, mode: str, stage_dir: Path) -> RestorationResult:
        if not self.repo_root.exists():
            raise FileNotFoundError(f"BOPTL repo not found: {self.repo_root}")
        stage_dir.mkdir(parents=True, exist_ok=True)

        # The upstream repo shells out via string-built commands on Windows, so
        # its input/output paths must avoid spaces.
        temp_work_dir = Path(tempfile.mkdtemp(prefix="photo_reviver_boptl_")).resolve()
        input_dir = temp_work_dir / "input"
        output_dir = temp_work_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        copied_input = input_dir / input_path.name
        shutil.copy2(input_path, copied_input)

        command = build_boptl_command(
            python_executable=self.python_executable,
            checkpoint_name=self.checkpoint_name,
            input_dir=input_dir,
            output_dir=output_dir,
            gpu=self.gpu,
            mode=mode,
        )

        log_path = (stage_dir / "boptl.log").resolve()
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=self.repo_root,
            )
            log_text = completed.stdout
            if completed.stderr:
                log_text = f"{log_text}\n\n[stderr]\n{completed.stderr}".strip()
            log_path.write_text(log_text, encoding="utf-8")
        except subprocess.CalledProcessError as error:
            log_text = error.stdout or ""
            if error.stderr:
                log_text = f"{log_text}\n\n[stderr]\n{error.stderr}".strip()
            log_path.write_text(log_text, encoding="utf-8")
            raise RuntimeError(
                "BOPTL restoration failed. Check the runtime log for details: "
                f"{log_path}"
            ) from error

        final_output_dir = output_dir / "final_output"
        output_files = sorted(path for path in final_output_dir.iterdir() if path.is_file())
        if not output_files:
            raise FileNotFoundError(
                "BOPTL finished, but no files were found in "
                f"{final_output_dir}"
            )

        restored_image = load_image(output_files[0])
        output_path = save_image(stage_dir / "restored_model_output.png", restored_image)

        result = RestorationResult(
            output_path=output_path.resolve(),
            backend=self.backend,
            notes=[
                "Restoration was delegated to the Bringing-Old-Photos-Back-to-Life repo.",
                f"Mode '{mode}' was mapped to the repo command flags.",
            ],
            command=command,
            log_path=log_path,
        )
        save_json(stage_dir / "restoration.json", result)
        return result


def build_restoration_runner(restoration_config: dict):
    backend = restoration_config.get("backend", "passthrough")
    if backend == "passthrough":
        return PassthroughRestorationRunner()
    if backend == "boptl":
        return BoptlRestorationRunner(
            repo_root=restoration_config["repo_root"],
            python_executable=restoration_config.get("python_executable", "python"),
            gpu=str(restoration_config.get("gpu", "-1")),
            checkpoint_name=restoration_config.get(
                "checkpoint_name",
                "Setting_9_epoch_100",
            ),
        )
    if backend == "external_command":
        command_template = restoration_config.get("external_command")
        if not command_template:
            raise ValueError("Missing 'external_command' template in config.")
        return ExternalCommandRestorationRunner(command_template)

    raise ValueError(f"Unsupported restoration backend: {backend}")
