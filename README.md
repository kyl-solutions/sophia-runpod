# Sophia RunPod — ACE-Step 1.5 Cover Mode

Serverless GPU worker for [Sophia](https://github.com/kyl-solutions/musicgen-vst) — AI music companion for musicians.

Runs **ACE-Step 1.5 Cover Mode** on RunPod serverless. Drop in reference audio, get back a stylistically transformed version with controllable fidelity.

## What It Does

- Receives base64-encoded audio + style parameters
- Runs ACE-Step 1.5 cover mode inference on GPU
- Returns base64-encoded WAV output
- Cover mode uses only the DiT model (~4GB VRAM) — no LLM needed

## Deploy on RunPod

1. Go to [runpod.io](https://runpod.io) → **Serverless** → **New Endpoint**
2. Select **Custom Template** → point to this repo's Dockerfile
3. GPU: **A40 48GB** (budget) or **A100 80GB** (fast)
4. Min workers: `0`, Max workers: `1`
5. Copy your **Endpoint ID**

## Input Schema

```json
{
  "input": {
    "reference_audio": "<base64 WAV>",
    "prompt": "gospel soul keys, warm Rhodes chords",
    "lyrics": "[Instrumental]",
    "audio_cover_strength": 0.5,
    "inference_steps": 8,
    "bpm": 92,
    "key_scale": "C Major",
    "duration": 30,
    "seed": -1,
    "shift": 3.0,
    "batch_size": 1
  }
}
```

## Key Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `audio_cover_strength` | 0.0 – 1.0 | 0 = free transform, 1 = strict preservation |
| `inference_steps` | 8 (turbo) | More steps = higher quality, slower |
| `shift` | 1.0 – 5.0 | Creativity vs semantic adherence |

## Part of Sophia

This is the GPU backend for [Sophia](https://github.com/kyl-solutions/musicgen-vst), a standalone macOS app for musicians. Drop audio from any DAW → AI generates stems + MIDI → drag back into your DAW.
