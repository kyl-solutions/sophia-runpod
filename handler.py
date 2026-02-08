"""RunPod serverless handler for ACE-Step 1.5 Cover Mode (v7).

v7: Diagnostic build — echo handler first to verify RunPod queue connectivity,
then model loading + full generation if echo works.

Two-phase startup:
  Phase 1: Register handler immediately (proves queue works)
  Phase 2: Load model on first real job (not echo/ping)
"""

import base64
import os
import sys
import tempfile
import time
import traceback

import runpod

# ──────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────

_dit_handler = None
_llm_handler = None
_model_loaded = False
_model_config = os.environ.get("ACESTEP_MODEL", "acestep-v15-turbo")


def _load_model():
    """Load model onto GPU. Called once on first real job."""
    global _dit_handler, _llm_handler, _model_loaded

    if _model_loaded:
        return True

    acestep_root = os.environ.get("ACESTEP_ROOT", "/app/acestep")

    print("[SOPHIA] ════════════════════════════════════════", flush=True)
    print("[SOPHIA] Loading ACE-Step 1.5 model...", flush=True)
    print(f"[SOPHIA]   config: {_model_config}", flush=True)
    print(f"[SOPHIA]   root:   {acestep_root}", flush=True)
    print(f"[SOPHIA]   device: cuda", flush=True)

    if os.path.exists(acestep_root):
        contents = os.listdir(acestep_root)
        print(f"[SOPHIA]   files:  {contents[:15]}", flush=True)
    else:
        print(f"[SOPHIA]   ERROR: root dir missing!", flush=True)
        return False

    start = time.time()

    try:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        print("[SOPHIA] Creating AceStepHandler...", flush=True)
        _dit_handler = AceStepHandler()

        print("[SOPHIA] Calling initialize_service...", flush=True)
        status_msg, success = _dit_handler.initialize_service(
            project_root=acestep_root,
            config_path=_model_config,
            device="cuda",
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
        )

        print(f"[SOPHIA] DiT result: success={success}", flush=True)
        print(f"[SOPHIA] DiT msg: {status_msg}", flush=True)

        if not success:
            print("[SOPHIA] FATAL: DiT init failed!", flush=True)
            return False

        _llm_handler = LLMHandler()
        _model_loaded = True

        elapsed = time.time() - start
        print(f"[SOPHIA] Model loaded in {elapsed:.1f}s", flush=True)
        print("[SOPHIA] ════════════════════════════════════════", flush=True)
        return True

    except Exception as e:
        print(f"[SOPHIA] Model load error: {e}", flush=True)
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────
# Handler — called per job
# ──────────────────────────────────────────────────────────

def handler(job):
    """RunPod serverless handler for ACE-Step 1.5 cover mode generation."""
    input_data = job["input"]

    # ── Echo/ping mode — test queue connectivity ──
    if input_data.get("ping"):
        print("[SOPHIA] PING received — queue is working!", flush=True)
        return {
            "pong": True,
            "model_loaded": _model_loaded,
            "timestamp": time.time(),
            "version": "v7",
        }

    # ── Load model on first real job ──
    if not _model_loaded:
        print("[SOPHIA] First real job — loading model...", flush=True)
        if not _load_model():
            return {"error": "Model failed to load"}

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
    print(f"[SOPHIA] Job received: strength={audio_cover_strength} "
          f"steps={inference_steps} bpm={bpm} key={key_scale} "
          f"duration={duration}s ref_size={file_size}B", flush=True)

    try:
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
        print(f"[SOPHIA] Generation done in {elapsed:.1f}s", flush=True)

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
        print(f"[SOPHIA] Output: {output_size} bytes, {elapsed:.1f}s", flush=True)

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
        try:
            os.unlink(ref_path)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────
# Entry point — register immediately, no pre-loading
# ──────────────────────────────────────────────────────────

print("[SOPHIA] v7 handler starting...", flush=True)
print("[SOPHIA] Registering with RunPod queue (no pre-load)...", flush=True)
runpod.serverless.start({"handler": handler})
