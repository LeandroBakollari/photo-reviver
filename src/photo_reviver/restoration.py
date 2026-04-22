from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from photo_reviver.io_utils import load_image, save_image, save_json
from photo_reviver.types import RestorationResult


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_subprocess_env(python_executable: str) -> dict[str, str] | None:
    executable_path = Path(python_executable)
    if not executable_path.parent or str(executable_path.parent) == ".":
        return None

    env = os.environ.copy()
    env["PATH"] = f"{executable_path.parent}{os.pathsep}{env.get('PATH', '')}"
    return env


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
                env=build_subprocess_env(self.python_executable),
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
            raise RuntimeError(self.build_failed_process_message(log_path)) from error

        final_output_dir = output_dir / "final_output"
        output_files = self.find_output_files(output_dir)
        if not output_files:
            raise RuntimeError(self.build_missing_output_message(final_output_dir, log_path, temp_work_dir))

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

    def build_failed_process_message(self, log_path: Path) -> str:
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if "ModuleNotFoundError: No module named 'torch'" in log_text:
            return (
                "BOPTL failed because the Python interpreter used by the Microsoft repo could not import torch. "
                f"The desktop app now launches BOPTL with this interpreter: {self.python_executable}. "
                "If this still fails, install the BOPTL dependencies into that environment. "
                f"Log: {log_path}"
            )
        return "BOPTL restoration failed. Check the runtime log for details: " f"{log_path}"

    def find_output_files(self, output_dir: Path) -> list[Path]:
        final_output_dir = output_dir / "final_output"
        if final_output_dir.exists():
            files = sorted(
                path
                for path in final_output_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if files:
                return files

        candidate_dirs = [
            output_dir / "stage_1_restore_output" / "restored_image",
            output_dir / "stage_3_face_output",
        ]
        candidates: list[Path] = []
        for directory in candidate_dirs:
            if directory.exists():
                candidates.extend(
                    path
                    for path in directory.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
        return sorted(candidates)

    def build_missing_output_message(
        self,
        final_output_dir: Path,
        log_path: Path,
        temp_work_dir: Path,
    ) -> str:
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        if "not enough memory" in log_text or "DefaultCPUAllocator" in log_text:
            return (
                "BOPTL did not create a restored image because the upstream model ran out of CPU memory. "
                "The input will now be resized before BOPTL, but for this completed run you need to try again. "
                "You can also choose passthrough mode, use a smaller image, or run BOPTL with a CUDA GPU. "
                f"Log: {log_path}. Temporary output: {temp_work_dir}"
            )
        if "Skip " in log_text and "due to an error" in log_text:
            return (
                "BOPTL skipped the input image and produced no restored output. "
                f"Check the upstream log for the exact error: {log_path}. "
                f"Temporary output: {temp_work_dir}"
            )
        return (
            "BOPTL finished, but no restored image was found in "
            f"{final_output_dir}. Check the runtime log: {log_path}. "
            f"Temporary output: {temp_work_dir}"
        )


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
