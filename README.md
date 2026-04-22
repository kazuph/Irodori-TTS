# Irodori-TTS

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-500M-v2)
[![VoiceDesign](https://img.shields.io/badge/VoiceDesign-HuggingFace-orange)](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v2-Demo)
[![VoiceDesign Demo](https://img.shields.io/badge/VoiceDesign%20Demo-HuggingFace%20Space-red)](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v2-VoiceDesign-Demo)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

Training and inference code for **Irodori-TTS**, a Flow Matching-based Text-to-Speech model. The architecture and training design largely follow [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/), using [DACVAE](https://github.com/facebookresearch/dacvae) continuous latents as the generation target.

> [!IMPORTANT]
> `main` tracks the **v2** codebase and is intended for use with the **Irodori-TTS-500M-v2** and **Irodori-TTS-500M-v2-VoiceDesign** model releases.
> If you need the previous v1 code, use the `v1` tag.
> v1 and v2 checkpoints / preprocessing are not compatible across versions.
> The previous public v1 model is available at [Aratako/Irodori-TTS-500M](https://huggingface.co/Aratako/Irodori-TTS-500M).

For model weights and audio samples, please refer to the [base model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v2) and the [VoiceDesign model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign).

## Features

- **Flow Matching TTS**: Rectified Flow Diffusion Transformer (RF-DiT) over continuous DACVAE latents
- **Voice Cloning**: Zero-shot voice cloning from reference audio
- **Voice Design**: Caption-conditioned style control
- **Multi-GPU Training**: Distributed training via `uv run torchrun` with gradient accumulation, mixed precision (bf16), and W&B logging
- **PEFT LoRA Fine-Tuning**: Parameter-efficient adaptation with PEFT/LoRA for released checkpoints
- **Flexible Inference**: CLI, Gradio Web UI, and HuggingFace Hub checkpoint support

## Architecture

The v2 codebase supports two closely related checkpoint families:

1. **Base model (`Aratako/Irodori-TTS-500M-v2`)**:
   Text encoder + reference latent encoder + diffusion transformer. The reference latent encoder consumes patched DACVAE latents from reference audio for speaker/style conditioning.
2. **VoiceDesign model (`Aratako/Irodori-TTS-500M-v2-VoiceDesign`)**:
   Text encoder + caption encoder + diffusion transformer. The caption encoder consumes style-control text and the speaker/reference branch is disabled.

Shared building blocks:

1. **Text Encoder**: Token embeddings initialized from a pretrained LLM, followed by self-attention + SwiGLU transformer layers with RoPE
2. **Condition Encoder**: Either a reference latent encoder for the base model or a caption encoder for the VoiceDesign model
3. **Diffusion Transformer**: Joint-attention DiT blocks with Low-Rank AdaLN (timestep-conditioned adaptive layer normalization), half-RoPE, and SwiGLU MLPs

Audio is represented as continuous latent sequences via the codec configured by the checkpoint. v2 uses the 32-dim [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) codec for 48kHz waveform reconstruction.

## Installation

```bash
git clone https://github.com/Aratako/Irodori-TTS.git
cd Irodori-TTS
uv sync
```

**Note**: For Linux/Windows with CUDA, PyTorch is automatically installed from the cu128 index. For macOS (MPS) or CPU-only usage, `uv sync` will install the default PyTorch build.

On macOS, the CLI now auto-selects `fp16` for `mps` when precision is not specified, and the Gradio UIs default to `mps + fp16` while keeping the DACVAE codec transient between requests to reduce cached-memory pressure.

On this fork, inference also supports `fp16` on `mps`. `bf16` remains CUDA-only.

## Why this fork exists

This fork exists to make **macOS MPS inference with `fp16`** usable for real local TTS workflows.

Upstream currently rejects `fp16` in `resolve_runtime_dtype()` unless the device is CUDA, so
`model_device=mps` + `model_precision=fp16` fails before inference starts. This fork adds:

- `fp16` support on `mps` in `resolve_runtime_dtype()`
- `fp16` exposure in `list_available_runtime_precisions()` for `mps`
- README documentation for the new supported precision/device combination

### Current macOS behavior

The current fork defaults to the following behavior on Apple Silicon:

- **CLI**: if `--model-precision` / `--codec-precision` are omitted, `mps` now auto-selects `fp16`
- **Gradio**: both UIs default to `mps + fp16`
- **Cached runtime**: the text-to-latent model stays cached, but the DACVAE codec is loaded only for active requests and released afterward

### Current Apple Silicon guidance

For local macOS inference, the current recommended setup is:

1. `model_device=mps`
2. `model_precision=fp16`
3. `codec_device=mps`
4. `codec_precision=fp16`
5. `decode_mode=sequential`

### Current memory status on Apple Silicon

Measured locally with the `Aratako/Irodori-TTS-500M-v2-VoiceDesign` checkpoint on an 8-second no-reference VoiceDesign request.

| Scenario | `torch.mps.driver_allocated_memory()` after load | after request | Summary |
|---|---:|---:|---|
| Legacy cached `fp32` path | 3.60 GB | 4.79 GB | Higher steady-state memory pressure |
| Current cached `fp16` + transient codec path | 2.41 GB | 2.53 GB | Lower cached-memory pressure while keeping MPS inference |

Practical takeaway:

- **`fp16` on MPS is the correct default** for this fork.
- **Keeping the codec transient matters more than moving it to CPU** for this workload.
- **This improves cached-memory behavior materially**, but it is still a PyTorch/MPS implementation rather than a full MLX rewrite or quantized deployment.

## Quick Start

### Simple Inference

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

### Inference without Reference Audio

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --no-ref \
  --output-wav outputs/sample.wav
```

### VoiceDesign Inference

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
  --text "今日はいい天気ですね。" \
  --caption "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

### Gradio Web UI

```bash
uv run python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

Then access the UI at `http://localhost:7860`.
The hosted v2 demo is available at [Aratako/Irodori-TTS-500M-v2-Demo](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v2-Demo).

For the VoiceDesign checkpoint, use the dedicated UI:

```bash
uv run python gradio_app_voicedesign.py --server-name 0.0.0.0 --server-port 7861
```

The hosted VoiceDesign demo is available at [Aratako/Irodori-TTS-500M-v2-VoiceDesign-Demo](https://huggingface.co/spaces/Aratako/Irodori-TTS-500M-v2-VoiceDesign-Demo).

`gradio_app.py` is for `Aratako/Irodori-TTS-500M-v2`. `gradio_app_voicedesign.py` is for `Aratako/Irodori-TTS-500M-v2-VoiceDesign`.

## Inference

### CLI

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2 \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

Local checkpoints (`.pt` or `.safetensors`) are also supported:

```bash
uv run python infer.py \
  --checkpoint outputs/checkpoint_final.safetensors \
  --text "今日はいい天気ですね。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

VoiceDesign checkpoints also support caption conditioning:

```bash
uv run python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-500M-v2-VoiceDesign \
  --text "今日はいい天気ですね。" \
  --caption "落ち着いた、近い距離感の女性話者" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` / `--hf-checkpoint` | (required, either one) | Local checkpoint file or Hugging Face repo id |
| `--text` | (required) | Text to synthesize |
| `--caption` | None | Optional style-control text for VoiceDesign checkpoints |
| `--output-wav` | `output.wav` | Output waveform path |
| `--ref-wav` | None | Reference waveform path for speaker conditioning |
| `--ref-latent` | None | Pre-computed reference latent (`.pt`) for speaker conditioning |
| `--no-ref` | False | Disable speaker reference conditioning |
| `--max-ref-seconds` | `30.0` | Maximum reference duration in seconds |
| `--ref-normalize-db` | -16.0 | Reference loudness target before DACVAE encode (set `none` to disable) |
| `--ref-ensure-max` | True | Scale reference down only when peak exceeds 1.0 (used when `--ref-normalize-db` is disabled) |
| `--codec-repo` | `Aratako/Semantic-DACVAE-Japanese-32dim` | Codec repo used for latent encode/decode |
| `--codec-deterministic-encode` | True | Use deterministic DACVAE encode path |
| `--codec-deterministic-decode` | True | Use deterministic DACVAE watermark-message decode path |
| `--enable-watermark` | False | Enable DACVAE watermark branch during decode |
| `--max-text-len` | checkpoint metadata or `256` | Maximum token length for text conditioning |
| `--max-caption-len` | checkpoint metadata or `max_text_len` | Maximum token length for caption conditioning |
| `--num-steps` | 40 | Number of Euler integration steps |
| `--num-candidates` | 1 | Number of candidates to generate in one pass |
| `--decode-mode` | `sequential` | Codec decode mode: `sequential` or `batch` |
| `--cfg-scale-text` | 3.0 | CFG scale for text conditioning |
| `--cfg-scale-caption` | 3.0 | CFG scale for caption conditioning |
| `--cfg-scale-speaker` | 5.0 | CFG scale for speaker conditioning |
| `--cfg-guidance-mode` | `independent` | CFG mode: `independent`, `joint`, `alternating` |
| `--cfg-scale` | None | Deprecated shared CFG override for all enabled conditions |
| `--cfg-min-t` | `0.5` | Lower timestep bound for CFG |
| `--cfg-max-t` | `1.0` | Upper timestep bound for CFG |
| `--truncation-factor` | None | Scale initial Gaussian noise before sampling |
| `--rescale-k` / `--rescale-sigma` | None | Temporal score rescaling parameters; must be set together |
| `--context-kv-cache` | True | Precompute context K/V projections for faster sampling |
| `--speaker-kv-scale` | None | Extra speaker K/V scaling for stronger speaker identity |
| `--speaker-kv-min-t` | `0.9` | Disable speaker K/V scaling after this timestep threshold |
| `--speaker-kv-max-layers` | None | Apply speaker K/V scaling only to first N diffusion layers |
| `--model-device` | auto | Device for model (`cuda`, `mps`, `cpu`) |
| `--codec-device` | auto | Device for DACVAE codec |
| `--model-precision` | best supported for device | Model precision (`fp32`, `bf16`, `fp16`). `bf16` is CUDA-only; `fp16` works on CUDA and MPS. On macOS `mps`, the default is `fp16`. |
| `--codec-precision` | best supported for device | Codec precision (`fp32`, `bf16`, `fp16`). `bf16` is CUDA-only; `fp16` works on CUDA and MPS. On macOS `mps`, the default is `fp16`. |
| `--seed` | random | Random seed for reproducibility |
| `--compile-model` | False | Enable `torch.compile` for faster inference |
| `--compile-dynamic` | False | Use `dynamic=True` for `torch.compile` |
| `--trim-tail` | True | Trim trailing silence via flattening heuristic |
| `--tail-window-size` | `20` | Window size used for tail trimming |
| `--tail-std-threshold` | `0.05` | Std threshold for tail trimming |
| `--tail-mean-threshold` | `0.1` | Mean threshold for tail trimming |
| `--show-timings` | True | Print per-stage timing breakdown |

## Training

### 1. Prepare Manifest (Precompute DACVAE Latents)

Encodes audio from a Hugging Face dataset into DACVAE latents and produces a JSONL manifest for training.

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

To include `speaker_id` in the manifest (for speaker-conditioned training):

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --speaker-column speaker \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

To include `caption` in the manifest (for caption-conditioned voice design training):

```bash
uv run python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --caption-column caption \
  --speaker-column speaker \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

When training the caption-conditioned voice-design model, `speaker_id` is optional. The
voice-design path disables speaker/reference conditioning and learns from `text + caption`.

This produces a JSONL manifest with entries like:

```json
{"text": "こんにちは", "caption": "落ち着いた、近い距離感の女性話者", "latent_path": "data/latents/00001.pt", "speaker_id": "myorg/my_dataset:speaker_001", "num_frames": 750}
```

### 2. Training

Single-GPU training:

```bash
uv run python train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts
```

VoiceDesign training uses a dedicated config:

```bash
uv run python train.py \
  --config configs/train_500m_v2_voice_design.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_voice_design
```

`configs/train_500m_v2_voice_design.yaml` sets `use_caption_condition: true` and disables the
speaker/reference branch. Caption-free configs continue to use speaker conditioning when
`speaker_id` / reference inputs are available.

The VoiceDesign config also enables `caption_warmup: true` for optional caption-branch warmup.
`warmup_steps` controls the LR scheduler, while `caption_warmup_steps` controls how long
non-caption gradients are discarded before normal joint training resumes.

Multi-GPU DDP training:

```bash
uv run torchrun --nproc_per_node 4 train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --device cuda
```

Training supports YAML config files with `model` and `train` sections. CLI arguments take precedence over YAML values. See `uv run python train.py --help` for all available options.

#### Fine-Tuning from Released Weights

Start a new training run from released inference weights (`.safetensors`). This initializes only the model weights; optimizer / scheduler state starts fresh.

```bash
uv run python train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_ft \
  --init-checkpoint path/to/Irodori-TTS-500M-v2.safetensors
```

LoRA fine-tuning:

```bash
uv run python train.py \
  --config configs/train_500m_v2_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_lora \
  --init-checkpoint path/to/Irodori-TTS-500M-v2.safetensors
```

Caption-conditioned voice-design LoRA fine-tuning:

```bash
uv run python train.py \
  --config configs/train_500m_v2_voice_design_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_voice_design_lora \
  --init-checkpoint path/to/Irodori-TTS-500M-v2.safetensors
```

Available LoRA target presets:

- `text_attn_mlp`: text encoder attention + attention gate + MLP
- `caption_attn_mlp`: caption encoder attention + attention gate + MLP
- `speaker_attn_mlp`: speaker encoder attention + attention gate + MLP, plus `speaker_encoder.in_proj`
- `diffusion_attn`: diffusion attention only, including text/speaker/caption context KV and attention gate
- `diffusion_attn_mlp`: `diffusion_attn` + diffusion MLP
- `all_attn`: all attention blocks across text/caption/speaker/diffusion, including attention gates
- `diffusion_full`: diffusion stack broadly: `cond_module`, `in_proj/out_proj`, diffusion attention, diffusion MLP, and AdaLN
- `adaln`: diffusion-block AdaLN layers only
- `conditioning`: conditioning-side projections only: `cond_module`, `speaker_encoder.in_proj`, and diffusion context KV projections
- `all_attn_mlp`: `all_attn` + text/caption/speaker/diffusion MLP, plus `speaker_encoder.in_proj`
- `all_linear`: all `nn.Linear` layers in the model; embeddings and norm weights are not included

`--lora-target-modules` also accepts a regex string or a comma-separated list of module suffixes. Resume automatically restores the saved LoRA config from the training checkpoint unless you explicitly override it.

When `--lora` is enabled, checkpoints are saved as adapter-only directories containing PEFT adapter weights plus trainer state for resume.

#### Resuming Interrupted Training

Resume an existing training run from a training checkpoint. Full-model runs use `.pt`; LoRA runs use checkpoint directories. Both restore optimizer, scheduler, and step state.

```bash
uv run python train.py \
  --config configs/train_500m_v2.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --resume outputs/irodori_tts/checkpoint_0010000.pt
```

LoRA resume example:

```bash
uv run python train.py \
  --config configs/train_500m_v2_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_lora \
  --resume outputs/irodori_tts_lora/checkpoint_0010000
```

If you move a LoRA checkpoint to another environment and the original base-checkpoint path is no longer valid, pass `--init-checkpoint path/to/base_model.safetensors` together with `--resume` to override the saved base-model path.

### 3. Checkpoint Conversion

Convert a training checkpoint to inference-only safetensors format:

```bash
uv run python convert_checkpoint_to_safetensors.py outputs/checkpoint_final.pt
```

LoRA adapter checkpoints can also be converted directly:

```bash
uv run python convert_checkpoint_to_safetensors.py outputs/irodori_tts_lora/checkpoint_final
```

LoRA adapter checkpoints are merged into the base model automatically during conversion, so the exported `.safetensors` file is directly usable for inference.

## Project Structure

```text
Irodori-TTS/
├── train.py                    # Training entry point (DDP support)
├── infer.py                    # CLI inference
├── gradio_app.py               # Gradio web UI
├── gradio_app_voicedesign.py   # Gradio web UI for VoiceDesign checkpoints
├── prepare_manifest.py         # Dataset -> DACVAE latent preprocessing
├── convert_checkpoint_to_safetensors.py  # Checkpoint converter
│
├── irodori_tts/                # Core library
│   ├── model.py                # TextToLatentRFDiT architecture
│   ├── rf.py                   # Rectified Flow utilities & Euler CFG sampling
│   ├── codec.py                # DACVAE codec wrapper
│   ├── dataset.py              # Dataset and collator
│   ├── tokenizer.py            # Pretrained LLM tokenizer wrapper
│   ├── config.py               # Model / Train / Sampling config dataclasses
│   ├── inference_runtime.py    # Cached, thread-safe inference runtime
│   ├── lora.py                 # PEFT LoRA integration helpers
│   ├── text_normalization.py   # Japanese text normalization
│   ├── optim.py                # Muon + AdamW optimizer
│   └── progress.py             # Training progress tracker
│
└── configs/
    ├── train_500m_v2.yaml                    # 500M v2 model config
    ├── train_500m_v2_lora.yaml               # 500M v2 LoRA fine-tuning config
    ├── train_500m_v2_voice_design.yaml       # 500M v2 VoiceDesign full fine-tuning config
    ├── train_500m_v2_voice_design_lora.yaml  # 500M v2 VoiceDesign LoRA fine-tuning config
    ├── train_500m.yaml                       # 500M v1 model config
    └── train_2.5b.yaml                       # 2.5B parameter model config
```

## License

- **Code**: [MIT License](LICENSE)
- **Model Weights**: Please refer to the [base model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v2) and the [VoiceDesign model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign) for licensing details

## Acknowledgments

This project builds upon the following works:

- [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/) — Architecture and training design reference
- [DACVAE](https://github.com/facebookresearch/dacvae) — Audio VAE

## Citation

```bibtex
@misc{irodori-tts,
  author = {Chihiro Arata},
  title = {Irodori-TTS: A Flow Matching-based Text-to-Speech Model with Emoji-driven Style Control},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Aratako/Irodori-TTS}}
}
```
