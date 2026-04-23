#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIN_DURATION_SECONDS = 0.5
MIN_RMS = 1e-3
MIN_PEAK = 1e-2
DEFAULT_MODEL_REPO = "mlx-community/Irodori-TTS-500M-v2-VoiceDesign-8bit"
DEFAULT_CFG_GUIDANCE_MODE = "independent"
DEFAULT_SEQUENCE_LENGTH = 750
DEFAULT_VENV_DIR = ".venv-mlx-audio"
MLX_AUDIO_GIT = "git+https://github.com/Blaizzy/mlx-audio"


@dataclass(slots=True)
class MlxRunRequest:
    text: str
    caption: str
    output_wav: str
    mlx_model: str = DEFAULT_MODEL_REPO
    cfg_guidance_mode: str = DEFAULT_CFG_GUIDANCE_MODE
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    cfg_scale: float | None = None
    ddpm_steps: int | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_runner_python() -> Path:
    return _repo_root() / DEFAULT_VENV_DIR / "bin" / "python"


def _has_irodori_backend(python_executable: str) -> bool:
    if not Path(python_executable).exists():
        return False
    probe = (
        "import importlib.util, sys; "
        "sys.exit(0 if importlib.util.find_spec('mlx_audio.tts.models.irodori_tts') else 1)"
    )
    result = subprocess.run(
        [python_executable, "-c", probe],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _run_bootstrap_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _bootstrap_runner(python_executable: str, runner_python: Path) -> None:
    runner_dir = runner_python.parent.parent
    if not runner_python.exists():
        print(f"[mlx-bootstrap] creating virtualenv at {runner_dir}", flush=True)
        created = _run_bootstrap_command([python_executable, "-m", "venv", str(runner_dir)])
        if created.stdout:
            print(created.stdout, end="", flush=True)
    print("[mlx-bootstrap] installing latest mlx-audio", flush=True)
    upgraded = _run_bootstrap_command(
        [str(runner_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    )
    if upgraded.stdout:
        print(upgraded.stdout, end="", flush=True)
    installed = _run_bootstrap_command(
        [str(runner_python), "-m", "pip", "install", "--upgrade", MLX_AUDIO_GIT]
    )
    if installed.stdout:
        print(installed.stdout, end="", flush=True)
    if not _has_irodori_backend(str(runner_python)):
        raise RuntimeError("mlx-audio installation completed, but Irodori-TTS backend is still unavailable.")


def ensure_mlx_runner(
    *,
    auto_bootstrap: bool = False,
    refresh: bool = False,
    python_executable: str | None = None,
) -> Path:
    current_python = Path(python_executable or sys.executable)
    runner_python = _default_runner_python()

    if refresh:
        _bootstrap_runner(str(current_python), runner_python)
        return runner_python

    if _has_irodori_backend(str(runner_python)):
        return runner_python

    if _has_irodori_backend(str(current_python)):
        return current_python

    if auto_bootstrap:
        _bootstrap_runner(str(current_python), runner_python)
        return runner_python

    raise RuntimeError(
        "No working mlx-audio environment with the Irodori-TTS backend was found.\n"
        "Run again with --bootstrap to create a dedicated local MLX environment."
    )


def _analyze_output(path: Path) -> None:
    import wave

    import numpy as np

    try:
        import soundfile as sf

        audio, sample_rate = sf.read(path)
        mono = audio.mean(axis=1) if getattr(audio, "ndim", 1) > 1 else audio
    except ModuleNotFoundError as exc:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
        dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
        if sample_width not in dtype_map:
            raise RuntimeError(
                f"Unsupported WAV sample width {sample_width} bytes without soundfile installed."
            ) from exc
        audio = np.frombuffer(frames, dtype=dtype_map[sample_width])
        if sample_width == 1:
            audio = (audio.astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32) / 2147483648.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        mono = audio
    duration = len(mono) / float(sample_rate)
    rms = float(np.sqrt(np.mean(np.square(mono))))
    peak = float(np.max(np.abs(mono)))
    print(
        f"[mlx-analysis] path={path} sr={sample_rate} duration={duration:.3f}s rms={rms:.6f} peak={peak:.6f}",
        flush=True,
    )
    if duration < MIN_DURATION_SECONDS:
        raise RuntimeError(f"Generated audio is too short: {duration:.3f}s")
    if rms < MIN_RMS or peak < MIN_PEAK:
        raise RuntimeError(
            "Generated audio looks too silent. Refusing to treat this output as valid."
        )


def _write_pcm_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import wave

    audio = np.clip(audio, -1.0, 1.0)
    if audio.ndim > 1:
        audio = audio.reshape(-1)
    pcm = np.clip(np.round(audio * 32767.0), -32768, 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _run_internal(args: argparse.Namespace) -> None:
    if args.caption is None or str(args.caption).strip() == "":
        raise ValueError("This validated MLX entrypoint currently requires --caption.")

    from mlx_audio.tts import load as load_tts_model

    output_path = Path(args.output_wav).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[mlx] model={args.mlx_model}", flush=True)
    print(f"[mlx] output={output_path}", flush=True)
    model = load_tts_model(str(args.mlx_model))
    generate_kwargs: dict[str, object] = {
        "text": str(args.text),
        "caption": str(args.caption),
        "cfg_guidance_mode": str(args.cfg_guidance_mode),
        "sequence_length": int(args.sequence_length),
    }
    if args.cfg_scale is not None:
        generate_kwargs["cfg_scale"] = float(args.cfg_scale)
    if args.ddpm_steps is not None:
        generate_kwargs["num_steps"] = int(args.ddpm_steps)
    result = next(
        model.generate(**generate_kwargs)
    )
    audio = np.array(result.audio, dtype=np.float32)
    _write_pcm_wav(output_path, audio, sample_rate=int(result.sample_rate))
    print(f"Duration:              {result.audio_duration}", flush=True)
    print(
        f"Samples/sec:           {result.audio_samples['samples-per-sec']:.1f}",
        flush=True,
    )
    print(
        f"Prompt:                {result.prompt['tokens']} tokens, {result.prompt['tokens-per-sec']:.1f} tokens-per-sec",
        flush=True,
    )
    print(
        f"Audio:                 {result.audio_samples['samples']} samples, {result.audio_samples['samples-per-sec']:.1f} samples-per-sec",
        flush=True,
    )
    print(f"Real-time factor:      {result.real_time_factor:.2f}x", flush=True)
    print(f"Processing time:       {result.processing_time_seconds:.2f}s", flush=True)
    print(f"Peak memory usage:     {result.peak_memory_usage:.2f}GB", flush=True)
    _analyze_output(output_path)


def build_external_command(request: MlxRunRequest, python_executable: Path) -> list[str]:
    cmd = [
        str(python_executable),
        str(Path(__file__).resolve()),
        "--internal-run",
        "--text",
        str(request.text),
        "--caption",
        str(request.caption),
        "--output-wav",
        str(request.output_wav),
        "--mlx-model",
        str(request.mlx_model),
        "--cfg-guidance-mode",
        str(request.cfg_guidance_mode),
        "--sequence-length",
        str(request.sequence_length),
    ]
    if request.cfg_scale is not None:
        cmd.extend(["--cfg-scale", str(request.cfg_scale)])
    if request.ddpm_steps is not None:
        cmd.extend(["--ddpm-steps", str(request.ddpm_steps)])
    return cmd


def run_mlx_generation(
    request: MlxRunRequest,
    *,
    bootstrap: bool = False,
    refresh: bool = False,
    capture_output: bool = False,
    python_executable: str | None = None,
) -> subprocess.CompletedProcess[str]:
    selected_python = ensure_mlx_runner(
        auto_bootstrap=bootstrap or refresh,
        refresh=refresh,
        python_executable=python_executable,
    )
    cmd = build_external_command(request, selected_python)
    env = os.environ.copy()
    env["IRODORI_MLX_INTERNAL"] = "1"
    return subprocess.run(
        cmd,
        check=True,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=capture_output,
        text=True,
    )


def _run_external(args: argparse.Namespace) -> None:
    request = MlxRunRequest(
        text=str(args.text),
        caption=str(args.caption),
        output_wav=str(args.output_wav),
        mlx_model=str(args.mlx_model),
        cfg_guidance_mode=str(args.cfg_guidance_mode),
        sequence_length=int(args.sequence_length),
        cfg_scale=None if args.cfg_scale is None else float(args.cfg_scale),
        ddpm_steps=None if args.ddpm_steps is None else int(args.ddpm_steps),
    )
    run_mlx_generation(request, refresh=bool(args.bootstrap))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validated MLX VoiceDesign runner for Apple Silicon."
    )
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--caption",
        required=True,
        help="VoiceDesign prompt. This validated MLX entrypoint is currently scoped to caption-driven synthesis.",
    )
    parser.add_argument("--output-wav", default="output_mlx.wav")
    parser.add_argument(
        "--mlx-model",
        default=DEFAULT_MODEL_REPO,
        help="MLX model repo or local path. Defaults to the validated 8bit VoiceDesign model.",
    )
    parser.add_argument(
        "--cfg-guidance-mode",
        choices=["independent", "joint", "alternating"],
        default=DEFAULT_CFG_GUIDANCE_MODE,
        help="Defaults to the upstream Irodori MLX sampler guidance mode.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Defaults to the upstream Irodori MLX sampler sequence length.",
    )
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--ddpm-steps", type=int, default=None)
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Create or refresh a dedicated local mlx-audio environment under .venv-mlx-audio before running.",
    )
    parser.add_argument(
        "--internal-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.internal_run or os.environ.get("IRODORI_MLX_INTERNAL") == "1":
        _run_internal(args)
    else:
        _run_external(args)


if __name__ == "__main__":
    main()
