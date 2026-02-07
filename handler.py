"""RunPod serverless handler for ACE-Step 1.5 Cover Mode (v3).

Runs on RunPod serverless (A100/A40/RTX 4090 GPU). Receives a job with:
  - reference_audio (base64 WAV)
  - prompt, lyrics, audio_cover_strength, inference_steps, etc.

Returns base64-encoded WAV audio.

Cover mode skips the LLM entirely — only the DiT model is needed (~4GB VRAM).
This means we can run on smaller GPUs (A40 48GB, or even RTX 4090 24GB).
"""

import base64
import os
import sys
import tempfile
import time
import traceback

import runpod

# ──────────────────────────────────────────────────────────
# Model loading — happens once at container startup
# ──────────────────────────────────────────────────────────

_dit_handler = None
_llm_handler = None
_model_config = os.environ.get("ACESTEP_MODEL", "acestep-v15-turbo")
_init_error = None  # Store init error for reporting


def _ensure_model_loaded():
    """Lazy-load model on first request (or at container startup)."""
    global _dit_handler, _llm_handler, _init_error

    if _dit_handler is not None:
        return

    if _init_error is not None:
        raise RuntimeError(f"Previous init failed: {_init_error}")

    acestep_root = os.environ.get("ACESTEP_ROOT", "/app/acestep")

    print(f"[RUNPOD] Loading ACE-Step 1.5 model...", flush=True)
    print(f"[RUNPOD]   config: {_model_config}", flush=True)
    print(f"[RUNPOD]   root: {acestep_root}", flush=True)
    print(f"[RUNPOD]   device: cuda", flush=True)

    # Debug: list what's in the root dir
    if os.path.exists(acestep_root):
        contents = os.listdir(acestep_root)
        print(f"[RUNPOD]   root contents: {contents[:20]}", flush=True)
    else:
        print(f"[RUNPOD]   WARNING: root dir does not exist!", flush=True)

    start = time.time()

    try:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        _dit_handler = AceStepHandler()
        status_msg, success = _dit_handler.initialize_service(
            project_root=acestep_root,
            config_path=_model_config,
            device="cuda",
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
        )

        print(f"[RUNPOD] DiT init result: success={success}, msg='{status_msg}'", flush=True)

        if not success:
            _dit_handler = None
            _init_error = f"DiT init failed: {status_msg}"
            raise RuntimeError(_init_error)

        # LLM handler — create but don't fully initialize (cover mode skips LLM)
        _llm_handler = LLMHandler()

        elapsed = time.time() - start
        print(f"[RUNPOD] Model loaded successfully in {elapsed:.1f}s", flush=True)

    except Exception as e:
        traceback.print_exc()
        _dit_handler = None
        _llm_handler = None
        _init_error = str(e)
        raise RuntimeError(f"Model init failed: {e}")


# ──────────────────────────────────────────────────────────
# Handler
# ──────────────────────────────────────────────────────────

def handler(event):
    """RunPod serverless handler for ACE-Step 1.5 cover mode generation."""
    try:
        _ensure_model_loaded()
    except Exception as e:
        return {"error": str(e)}

    if _dit_handler is None:
        return {"error": f"Model not loaded: {_init_error}"}

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

    try:
        # ── Run inference ──
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        params = GenerationParams(
            task_type="cover",
            reference_audio=ref_path,
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
            thinking=False,
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

        return {
            "audio_b64": audio_b64,
            "format": "wav",
            "duration": duration,
            "seed": seed,
            "inference_time": round(elapsed, 2),
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Inference error: {str(e)}"}

    finally:
        # Clean up
        try:
            os.unlink(ref_path)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Try to pre-load at startup, but don't crash if it fails —
    # the handler will retry and return errors per-job instead.
    try:
        _ensure_model_loaded()
        print("[RUNPOD] Startup model pre-load: SUCCESS", flush=True)
    except Exception as e:
        print(f"[RUNPOD] Startup model pre-load FAILED (will retry per-job): {e}", flush=True)

    # Always start the handler — even if model init failed
    runpod.serverless.start({"handler": handler})
