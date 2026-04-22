#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

import gradio as gr

from infer_mlx import (
    DEFAULT_CFG_GUIDANCE_MODE,
    DEFAULT_MODEL_REPO,
    DEFAULT_SEQUENCE_LENGTH,
    MlxRunRequest,
    ensure_mlx_runner,
    run_mlx_generation,
)

TIMING_PREFIXES = (
    "Duration:",
    "Samples/sec:",
    "Prompt:",
    "Audio:",
    "Real-time factor:",
    "Processing time:",
    "Peak memory usage:",
    "[mlx-postprocess]",
    "[mlx-analysis]",
)
MAX_HISTORY_ITEMS = 24
EMPTY_HISTORY_TEXT = "再生履歴はまだありません。"


@dataclass(frozen=True, slots=True)
class SamplePreset:
    key: str
    label: str
    text: str
    caption: str


SAMPLE_PRESETS: tuple[SamplePreset, ...] = (
    SamplePreset(
        key="calm-close",
        label="落ち着き女性",
        text="今日はいい天気ですね。ゆっくり散歩したくなります。",
        caption="落ち着いた、近い距離感の女性話者。柔らかく自然に話す。",
    ),
    SamplePreset(
        key="bright-morning",
        label="朝番組",
        text="おはようございます。今日も元気に一日を始めていきましょう。",
        caption="明るく爽やかな朝番組の女性ナレーター。はっきり前向きに読む。",
    ),
    SamplePreset(
        key="gentle-guide",
        label="案内音声",
        text="次の電車は三番線にまいります。足元にご注意ください。",
        caption="親切で聞き取りやすい公共案内の女性アナウンス。",
    ),
    SamplePreset(
        key="soft-whisper",
        label="ささやき",
        text="大丈夫、落ち着いて。ひとつずつやればちゃんと進めますよ。",
        caption="息遣いを少し含む、小さめで親密なささやき声の女性話者。",
    ),
    SamplePreset(
        key="energetic-vtuber",
        label="元気配信者",
        text="みなさんこんにちはー。今日は新しい企画をたっぷりお届けします。",
        caption="テンション高めで愛嬌のある女性配信者。勢いよく明るく話す。",
    ),
    SamplePreset(
        key="cool-narrator",
        label="クール紹介",
        text="静かな夜の街に、ひとすじの光が差し込んだ。",
        caption="クールで透明感のある女性ナレーター。抑制を保ちつつ印象的に読む。",
    ),
    SamplePreset(
        key="warm-story",
        label="朗読",
        text="小さな店の扉を開けると、懐かしい珈琲の香りが広がっていた。",
        caption="温かみのある朗読向きの女性話者。情景をやさしく描写する。",
    ),
    SamplePreset(
        key="business-male",
        label="落ち着き男性",
        text="本日の会議は十時から開始します。資料のご準備をお願いします。",
        caption="落ち着いた低めの男性話者。丁寧で信頼感のあるビジネストーン。",
    ),
    SamplePreset(
        key="trailer-male",
        label="予告編",
        text="その瞬間、運命は静かに動き始めた。",
        caption="重厚感のある男性ナレーター。映画予告のように引きを作る。",
    ),
    SamplePreset(
        key="cute-character",
        label="かわいい声",
        text="えへへ、ちゃんと見ててくださいね。今日はぜったい成功させます。",
        caption="少し高めでかわいらしい女性キャラクターボイス。愛嬌たっぷりに話す。",
    ),
    SamplePreset(
        key="radio-host",
        label="ラジオ司会",
        text="今夜も始まりました、深夜のリラックスタイム。最後までごゆっくり。",
        caption="聞き心地の良いラジオパーソナリティ。自然で滑らか、包み込むように話す。",
    ),
    SamplePreset(
        key="kids-book",
        label="絵本読み",
        text="そしてうさぎさんは、大きな月を見上げながらそっと目を閉じました。",
        caption="子ども向け読み聞かせに合う、やさしく表情豊かな女性話者。",
    ),
)


def _parse_optional_float(raw: str | None, label: str) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a float or blank.") from exc


def _parse_optional_int(raw: str | None, label: str) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an int or blank.") from exc


def _extract_timing_text(stdout: str) -> str:
    lines = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(TIMING_PREFIXES):
            lines.append(stripped)
    return "\n".join(lines) if lines else "No timing lines were emitted."


def _prepare_mlx_environment() -> str:
    runner_python = ensure_mlx_runner(auto_bootstrap=True, refresh=True)
    return f"Prepared dedicated MLX runner: {runner_python}"


def _normalize_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    if not history:
        return []
    normalized: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        audio_path = str(item.get("audio_path", "")).strip()
        if label == "" or audio_path == "":
            continue
        normalized.append(
            {
                "label": label,
                "audio_path": audio_path,
                "text": str(item.get("text", "")),
                "caption": str(item.get("caption", "")),
                "run_log": str(item.get("run_log", "")),
                "timing_text": str(item.get("timing_text", "")),
            }
        )
    return normalized[:MAX_HISTORY_ITEMS]


def _history_dropdown_update(history: list[dict[str, str]]) -> gr.Dropdown:
    labels = [item["label"] for item in history]
    return gr.Dropdown(choices=labels, value=labels[0] if labels else None)


def _history_markdown(history: list[dict[str, str]]) -> str:
    if not history:
        return EMPTY_HISTORY_TEXT
    lines = ["**再生履歴**"]
    for idx, item in enumerate(history, start=1):
        lines.append(f"{idx}. {item['label']}")
    return "\n".join(lines)


def _make_history_entry(
    *,
    text: str,
    caption: str,
    audio_path: str,
    run_log: str,
    timing_text: str,
) -> dict[str, str]:
    created_at = datetime.now().strftime("%H:%M:%S")
    short_text = text.strip().replace("\n", " ")
    if len(short_text) > 18:
        short_text = f"{short_text[:18]}..."
    return {
        "label": f"{created_at} | {short_text}",
        "audio_path": audio_path,
        "text": text,
        "caption": caption,
        "run_log": run_log,
        "timing_text": timing_text,
    }


def _push_history(
    history: list[dict[str, str]] | None,
    entry: dict[str, str],
) -> list[dict[str, str]]:
    normalized = _normalize_history(history)
    deduped = [item for item in normalized if item["audio_path"] != entry["audio_path"]]
    return [entry, *deduped][:MAX_HISTORY_ITEMS]


def _find_history_entry(
    history: list[dict[str, str]] | None,
    selected_label: str | None,
) -> dict[str, str] | None:
    normalized = _normalize_history(history)
    if not normalized:
        return None
    if selected_label is None or str(selected_label).strip() == "":
        return normalized[0]
    for item in normalized:
        if item["label"] == selected_label:
            return item
    return normalized[0]


def _run_generation(
    text: str,
    caption: str,
    mlx_model: str,
    cfg_guidance_mode: str,
    sequence_length: int,
    cfg_scale_raw: str,
    ddpm_steps_raw: str,
    bootstrap_if_missing: bool,
) -> tuple[str, str, str]:
    text_value = str(text).strip()
    caption_value = str(caption).strip()
    model_value = str(mlx_model).strip()

    if text_value == "":
        raise ValueError("text is required.")
    if caption_value == "":
        raise ValueError("caption is required for the validated MLX VoiceDesign path.")
    if model_value == "":
        raise ValueError("mlx_model is required.")

    out_dir = Path("gradio_outputs_mlx")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = out_dir / f"sample_{stamp}.wav"
    request = MlxRunRequest(
        text=text_value,
        caption=caption_value,
        output_wav=str(output_path),
        mlx_model=model_value,
        cfg_guidance_mode=str(cfg_guidance_mode),
        sequence_length=int(sequence_length),
        cfg_scale=_parse_optional_float(cfg_scale_raw, "cfg_scale"),
        ddpm_steps=_parse_optional_int(ddpm_steps_raw, "ddpm_steps"),
    )

    try:
        completed = run_mlx_generation(
            request,
            bootstrap=bool(bootstrap_if_missing),
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        log_parts = ["MLX generation failed."]
        if exc.stdout:
            log_parts.append(exc.stdout.strip())
        if exc.stderr:
            log_parts.append(exc.stderr.strip())
        raise RuntimeError("\n\n".join(part for part in log_parts if part)) from exc

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    log_lines = [f"saved: {output_path}"]
    if stdout:
        log_lines.append(stdout)
    if stderr:
        log_lines.append(f"[stderr]\n{stderr}")
    return str(output_path), "\n\n".join(log_lines), _extract_timing_text(stdout)


def _generate_with_history(
    history: list[dict[str, str]] | None,
    text: str,
    caption: str,
    mlx_model: str,
    cfg_guidance_mode: str,
    sequence_length: int,
    cfg_scale_raw: str,
    ddpm_steps_raw: str,
    bootstrap_if_missing: bool,
) -> tuple[str, str, str, list[dict[str, str]], gr.Dropdown, str, str, str, str]:
    audio_path, run_log, timing_text = _run_generation(
        text=text,
        caption=caption,
        mlx_model=mlx_model,
        cfg_guidance_mode=cfg_guidance_mode,
        sequence_length=sequence_length,
        cfg_scale_raw=cfg_scale_raw,
        ddpm_steps_raw=ddpm_steps_raw,
        bootstrap_if_missing=bootstrap_if_missing,
    )
    updated_history = _push_history(
        history,
        _make_history_entry(
            text=str(text),
            caption=str(caption),
            audio_path=audio_path,
            run_log=run_log,
            timing_text=timing_text,
        ),
    )
    return (
        audio_path,
        run_log,
        timing_text,
        updated_history,
        _history_dropdown_update(updated_history),
        _history_markdown(updated_history),
        audio_path,
        run_log,
        timing_text,
    )


def _run_sample_preset(
    preset: SamplePreset,
    history: list[dict[str, str]] | None,
    mlx_model: str,
    cfg_guidance_mode: str,
    sequence_length: int,
    cfg_scale_raw: str,
    ddpm_steps_raw: str,
    bootstrap_if_missing: bool,
) -> tuple[str, str, str, str, str, list[dict[str, str]], gr.Dropdown, str, str, str, str]:
    (
        audio_path,
        run_log,
        timing_text,
        updated_history,
        history_dropdown,
        history_summary,
        history_audio,
        history_log,
        history_timing,
    ) = _generate_with_history(
        history,
        text=preset.text,
        caption=preset.caption,
        mlx_model=mlx_model,
        cfg_guidance_mode=cfg_guidance_mode,
        sequence_length=sequence_length,
        cfg_scale_raw=cfg_scale_raw,
        ddpm_steps_raw=ddpm_steps_raw,
        bootstrap_if_missing=bootstrap_if_missing,
    )
    return (
        preset.text,
        preset.caption,
        audio_path,
        run_log,
        timing_text,
        updated_history,
        history_dropdown,
        history_summary,
        history_audio,
        history_log,
        history_timing,
    )


def _restore_history(
    history: list[dict[str, str]] | None,
) -> tuple[gr.Dropdown, str, str | None, str, str]:
    normalized = _normalize_history(history)
    current = normalized[0] if normalized else None
    return (
        _history_dropdown_update(normalized),
        _history_markdown(normalized),
        None if current is None else current["audio_path"],
        "" if current is None else current["run_log"],
        "" if current is None else current["timing_text"],
    )


def _select_history_item(
    selected_label: str | None,
    history: list[dict[str, str]] | None,
) -> tuple[str | None, str, str, str, str]:
    selected = _find_history_entry(history, selected_label)
    if selected is None:
        return None, "", "", "", ""
    return (
        selected["audio_path"],
        selected["run_log"],
        selected["timing_text"],
        selected["text"],
        selected["caption"],
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Irodori-TTS MLX VoiceDesign Gradio") as demo:
        gr.Markdown("# Irodori-TTS MLX VoiceDesign Playground")
        gr.Markdown(
            "Apple Silicon 向けの検証済み MLX 8bit VoiceDesign 経路です。"
            "生成後はすぐ再生し、再生済み音声はブラウザ内の履歴に残します。"
        )

        history_state = gr.BrowserState([], storage_key="irodori-tts-mlx-playback-history")

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                with gr.Group():
                    gr.Markdown("### Generate")
                    with gr.Row():
                        prepare_btn = gr.Button("Prepare / Refresh MLX Environment")
                        generate_btn = gr.Button("Generate", variant="primary")
                    prepare_status = gr.Textbox(
                        label="Environment Status",
                        interactive=False,
                        lines=2,
                    )
                    text = gr.Textbox(label="Text", lines=4)
                    caption = gr.Textbox(label="Caption / Style Prompt", lines=4)

            with gr.Column(scale=3):
                with gr.Group():
                    gr.Markdown("### Now Playing")
                    out_audio = gr.Audio(
                        label="Generated Audio",
                        type="filepath",
                        interactive=False,
                        autoplay=True,
                    )
                    out_timing = gr.Textbox(label="Latest Timing", lines=8, interactive=False)

            with gr.Column(scale=4):
                with gr.Group():
                    gr.Markdown("### Playback History")
                    history_choice = gr.Dropdown(
                        label="Previously Played",
                        choices=[],
                        value=None,
                        interactive=True,
                    )
                    history_audio = gr.Audio(
                        label="Replay Selected History",
                        type="filepath",
                        interactive=False,
                    )
                    history_timing = gr.Textbox(
                        label="Selected History Timing",
                        lines=8,
                        interactive=False,
                    )
                    history_summary = gr.Markdown(EMPTY_HISTORY_TEXT)

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                with gr.Group():
                    gr.Markdown("### One-click Samples")
                    gr.Markdown(
                        "下のボタンは **1回押すだけで** text / caption を入れてそのまま生成します。"
                    )
                    sample_buttons: list[gr.Button] = []
                    sample_cols = 4
                    for row_start in range(0, len(SAMPLE_PRESETS), sample_cols):
                        with gr.Row():
                            for preset in SAMPLE_PRESETS[row_start : row_start + sample_cols]:
                                sample_buttons.append(gr.Button(preset.label))

            with gr.Column(scale=3):
                with gr.Group():
                    gr.Markdown("### Settings")
                    mlx_model = gr.Textbox(label="MLX Model Repo", value=DEFAULT_MODEL_REPO)
                    cfg_guidance_mode = gr.Dropdown(
                        label="CFG Guidance Mode",
                        choices=["independent", "joint", "alternating"],
                        value=DEFAULT_CFG_GUIDANCE_MODE,
                    )
                    sequence_length = gr.Slider(
                        label="Sequence Length",
                        minimum=50,
                        maximum=1000,
                        value=DEFAULT_SEQUENCE_LENGTH,
                        step=10,
                    )
                    bootstrap_if_missing = gr.Checkbox(
                        label="Auto-bootstrap dedicated MLX environment if missing",
                        value=True,
                    )
                    cfg_scale_raw = gr.Textbox(label="CFG Scale Override (optional)", value="")
                    ddpm_steps_raw = gr.Textbox(label="DDPM Steps Override (optional)", value="")

            with gr.Column(scale=4):
                with gr.Group():
                    gr.Markdown("### Logs")
                    out_log = gr.Textbox(label="Latest Run Log", lines=12, interactive=False)
                    history_log = gr.Textbox(
                        label="Selected History Log",
                        lines=12,
                        interactive=False,
                    )

        prepare_btn.click(_prepare_mlx_environment, outputs=[prepare_status])
        generate_btn.click(
            _generate_with_history,
            inputs=[
                history_state,
                text,
                caption,
                mlx_model,
                cfg_guidance_mode,
                sequence_length,
                cfg_scale_raw,
                ddpm_steps_raw,
                bootstrap_if_missing,
            ],
            outputs=[
                out_audio,
                out_log,
                out_timing,
                history_state,
                history_choice,
                history_summary,
                history_audio,
                history_log,
                history_timing,
            ],
        )
        for preset, button in zip(SAMPLE_PRESETS, sample_buttons, strict=True):
            button.click(
                partial(_run_sample_preset, preset),
                inputs=[
                    history_state,
                    mlx_model,
                    cfg_guidance_mode,
                    sequence_length,
                    cfg_scale_raw,
                    ddpm_steps_raw,
                    bootstrap_if_missing,
                ],
                outputs=[
                    text,
                    caption,
                    out_audio,
                    out_log,
                    out_timing,
                    history_state,
                    history_choice,
                    history_summary,
                    history_audio,
                    history_log,
                    history_timing,
                ],
            )

        history_choice.change(
            _select_history_item,
            inputs=[history_choice, history_state],
            outputs=[history_audio, history_log, history_timing, text, caption],
        )
        demo.load(
            _restore_history,
            inputs=[history_state],
            outputs=[history_choice, history_summary, history_audio, history_log, history_timing],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio app for validated MLX VoiceDesign.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
    )


if __name__ == "__main__":
    main()
