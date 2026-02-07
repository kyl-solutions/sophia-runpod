"""RunPod serverless handler for ACE-Step 1.5 Cover Mode (v2).

Runs on RunPod serverless (A100/A40 GPU). Receives a job with:
  - reference_audio (base64 WAV)
  - prompt, lyrics, audio_cover_strength, inference_steps, etc.

Returns base64-encoded WAV audio.

Cover mode skips the LLM entirely — only the DiT model is needed (~4GB VRAM).
This means we can run on smaller GPUs (A40 48GB, or even RTX 4090 24GB).
"""

import base64
import os
import tempfile
import time
from typing import Optional

import runpod

# ──────────────────────────────────────────────────────────
# Model loading — happens once at container startup
# ──────────────────────────────────────────────────────────

_dit_handler = None
_llm_handler = None
_model_config = os.environ.get("ACESTEP_MODEL", "acestep-v15-turbo")


def _ensure_model_loaded():
    """Lazy-load model on first request (or at container startup)."""
    global _dit_handler, _llm_handler

    if _dit_handler is not None:
        return

    print(f"[RUNPOD] Loading ACE-Step 1.5 model: {_model_config}", flush=True)
    start = time.time()

    try:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        # ACE-Step repo is at /app/acestep — project_root must point there
        acestep_root = os.environ.get("ACESTEP_ROOT", "/app/acestep")
        print(f"[RUNPOD] ACE-Step root: {acestep_root}", flush=True)
        print(f"[RUNPOD] Config: {_model_config}", flush=True)

        _dit_handler = AceStepHandler()
        _dit_handler.initialize_service(
            project_root=acestep_root,
            config_path=_model_config,
            device="cuda",
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
        )

        # LLM handler — create but don't initialize (cover mode skips LLM)
        _llm_handler = LLMHandler()

        elapsed = time.time() - start
        print(f"[RUNPOD] Model loaded in {elapsed:.1f}s", flush=True)

    except Exception as e:
        import traceback
        print(f"[RUNPOD] FATAL: Model loading failed: {e}", flush=True)
        traceback.print_exc()
        _dit_handler = None
        _llm_handler = None
        raise RuntimeError(f"Model not fully initialized: {e}")


# ──────────────────────────────────────────────────────────
# Handler
# ──────────────────────────────────────────────────────────

def handler(event):
    """RunPod serverless handler for ACE-Step 1.5 cover mode generation.

    Input schema:
        reference_audio: str     — base64-encoded WAV/MP3/FLAC
        prompt: str              — style/genre description (caption)
        lyrics: str              — "[Instrumental]" or actual lyrics
        audio_cover_strength: float — 0.0 (free transform) to 1.0 (strict structure)
        inference_steps: int     — 8 for turbo, 50 for base/sft
        bpm: int | None          — tempo (30-300), None = auto
        key_scale: str           — e.g. "C Major", "Am", "" = auto
        duration: float          — audio duration in seconds (10-600)
        seed: int                — -1 = random
        shift: float             — 1.0-5.0, creativity vs semantic (default 3.0)
        batch_size: int          — number of variations (default 1)

    Returns:
        audio_b64: str           — base64-encoded WAV
        format: str              — "wav"
        duration: float          — actual output duration
        seed: int                — seed used (for reproducibility)
    """
    try:
        _ensure_model_loaded()
    except Exception as e:
        return {"error": str(e)}

    if _dit_handler is None:
        return {"error": "Model not fully initialized"}

    input_data = event["input"]

    # ── Extract parameters ──
    reference_audio_b64 = input_data.get("reference_audio", "")
    prompt = input_data.get("prompt", "")
    lyrics = input_data.get("lyrics", "[Instrumental]")
    audio_cover_strength = float(input_data.get("audio_cover_strength", 0.5))
    inference_steps = int(input_data.get("inference_steps", 8))
    bpm = input_data.get("bpm")
    if bpm is not None:
        bpm = int(bpm)
    key_scale = input_data.get("key_scale", "")
    duration = float(input_data.get("duration", 30))
    seed = int(input_data.get("seed", -1))
    shift = float(input_data.get("shift", 3.0))
    batch_size = int(input_data.get("batch_size", 1))

    if not reference_audio_b64:
        return {"error": "reference_audio (base64) is required for cover mode"}

    # ── Write reference audio to temp file ──
    ref_path = os.path.join(tempfile.gettempdir(), "sophia_ref_input.wav")
    with open(ref_path, "wb") as f:
        f.write(base64.b64decode(reference_audio_b64))

    file_size = os.path.getsize(ref_path)
    print(f"[RUNPOD] Cover mode: strength={audio_cover_strength} "
          f"steps={inference_steps} bpm={bpm} key={key_scale} "
          f"duration={duration}s ref_size={file_size}B", flush=True)

    # ── Run inference ──
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    params = GenerationParams(
        task_type="cover",
        src_audio=ref_path,
        audio_cover_strength=audio_cover_strength,
        caption=prompt,
        lyrics=lyrics,
        instrumental=lyrics.strip().lower() in ("[instrumental]", "[inst]", ""),
        bpm=bpm,
        keyscale=key_scale,
        duration=duration,
        inference_steps=inference_steps,
        seed=seed,
        shift=shift,
        thinking=False,  # LLM skipped for cover anyway
        infer_method="ode",
    )

    config = GenerationConfig(
        batch_size=batch_size,
        audio_format="wav",
        use_random_seed=(seed == -1),
    )

    save_dir = tempfile.mkdtemp(prefix="sophia_runpod_")
    start_time = time.time()

    result = generate_music(
        dit_handler=_dit_handler,
        llm_handler=_llm_handler,
        params=params,
        config=config,
        save_dir=save_dir,
    )

    elapsed = time.time() - start_time
    print(f"[RUNPOD] Generation completed in {elapsed:.1f}s", flush=True)

    if not result.success:
        return {"error": f"Generation failed: {result.error}"}

    if not result.audios:
        return {"error": "Generation produced no audio outputs"}

    # ── Return first result as base64 ──
    audio_info = result.audios[0]
    audio_path = audio_info.get("path", "")

    if not audio_path or not os.path.exists(audio_path):
        return {"error": f"Output audio not found at: {audio_path}"}

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    output_size = os.path.getsize(audio_path)
    print(f"[RUNPOD] Output: {audio_path} ({output_size} bytes, {elapsed:.1f}s)", flush=True)

    # Clean up
    try:
        os.unlink(ref_path)
    except OSError:
        pass

    return {
        "audio_b64": audio_b64,
        "format": "wav",
        "duration": duration,
        "seed": seed,
        "inference_time": round(elapsed, 2),
    }


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-load model at container startup
    _ensure_model_loaded()
    runpod.serverless.start({"handler": handler})
