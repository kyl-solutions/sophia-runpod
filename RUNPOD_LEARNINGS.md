# Sophia RunPod Deployment — Learnings & Next Steps

> **Date:** 2026-02-08
> **Status:** Handler works (ping returns in 60ms) but full generation fails on model load.
> **Priority:** Fix the 2-3 root causes below, then generation should work.

---

## Current State (v7.1 — commit `85029c7`)

### What Works
- ✅ Handler registers with RunPod queue correctly
- ✅ Ping mode (`{"input": {"ping": true}}`) returns `{"pong": true, "version": "v7"}` in 60ms
- ✅ Docker image builds successfully via GitHub Actions (~12 min)
- ✅ GraphQL API can update template image + endpoint config programmatically
- ✅ Workers do eventually become "ready" and pick up jobs

### What Fails
- ❌ Model loading inside handler returns "Model failed to load" (no detailed error yet)
- ❌ Cold starts take 10+ minutes (14.8GB image pull)
- ❌ Jobs submitted before worker is ready get stuck IN_QUEUE permanently
- ❌ Workers sometimes show "idle/ready" but don't process queued jobs

---

## Root Causes Identified (HIGH CONFIDENCE)

### 1. `RUNPOD_INIT_TIMEOUT` — Workers Killed Before Ready
**The #1 issue.** RunPod's default health check timeout is **7 minutes (420 seconds)**. Our 14.8GB image takes **10+ minutes** to pull from ghcr.io. Workers are being killed and restarted before they finish initializing, creating a crash loop.

**Fix:**
```
Set env var: RUNPOD_INIT_TIMEOUT=900
```
Via GraphQL:
```json
{
  "query": "mutation { saveTemplate(input: { id: \"g8bgahscs8\", name: \"sophia-v8\", imageName: \"ghcr.io/kyl-solutions/sophia-runpod:v-XXXXX\", containerDiskInGb: 40, volumeInGb: 0, dockerArgs: \"\", env: [{key: \"RUNPOD_INIT_TIMEOUT\", value: \"900\"}] }) { id } }"
}
```

**Source:** [RunPod Docs - Optimization](https://docs.runpod.io/serverless/development/optimization), [AnswerOverflow](https://www.answeroverflow.com/m/1201576784794222602)

### 2. RunPod SDK Routing Bug (v1.7.11 / v1.7.12)
RunPod Python SDK versions newer than 1.7.10 have a critical bug where concurrent requests route to the **same single worker**, causing massive queue delays.

**Fix:** Pin to `runpod==1.7.10` in the Dockerfile:
```dockerfile
RUN pip install --no-cache-dir runpod==1.7.10
```

**Source:** [GitHub Issue #432](https://github.com/runpod/runpod-python/issues/432)

### 3. ghcr.io Registry Issues
- ghcr.io images default to **private**. RunPod cannot pull without credentials.
- Even public ghcr.io images can be slower to pull than Docker Hub.
- RunPod infrastructure is optimized for Docker Hub.

**Fix options:**
- Ensure ghcr.io package is set to **public** in GitHub Package Settings
- OR switch to Docker Hub (`docker.io/kylsolutions/sophia-runpod`)
- OR use RunPod's GitHub Integration (builds on RunPod's infrastructure, no pull needed)

---

## Strategies Tried & Results

| Version | Strategy | Result |
|---------|----------|--------|
| v2 | Pre-load model, basic handler | Crash loop — wrong project_root |
| v3 | Fix crash loop, check init return | Workers stuck initializing forever |
| v4 | Lazy model load (register immediately) | Workers "ready" but jobs stuck IN_QUEUE |
| v5 | Add vector_quantize_pytorch dep | First real error: dependency missing → fixed |
| v6 | Pre-load model before start() | Workers init for 9 min, become ready, but don't pick up jobs |
| v7 | Module-level start(), ping mode, no pre-load | **PING WORKS** (60ms). Real jobs fail with "Model failed to load" |
| v7.1 | v7 + detailed error reporting | Workers killed by init timeout before becoming ready |

### Key Timeline Discovery
- v6: Workers `init=2` for 9 min → `idle=2, ready=2` → jobs stuck IN_QUEUE
- v7: Workers `init=1` for 10 min → `idle=1` → picked up ping in 60ms → real job failed
- The inconsistency is explained by `RUNPOD_INIT_TIMEOUT` — workers being killed and restarted at 7 min mark

---

## Architecture Recommendations for Next Session

### Immediate Fixes (Do These First)
1. **Set `RUNPOD_INIT_TIMEOUT=900`** as template env var
2. **Pin `runpod==1.7.10`** in Dockerfile
3. **Verify ghcr.io image is public** (or switch to Docker Hub)
4. **Move model loading BEFORE `runpod.serverless.start()`** — the docs say this is the correct pattern. Worker stays "initializing" during load, becomes "ready" only when done.

### Two-Dockerfile Strategy (Recommended)
Split into base + handler images to make iteration fast:

```dockerfile
# Dockerfile.base — build once, rarely changes
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04
# ... all system deps, torch, ACE-Step deps, model weights ...
# Tag: ghcr.io/kyl-solutions/sophia-runpod:base-1.0

# Dockerfile — iterates fast
FROM ghcr.io/kyl-solutions/sophia-runpod:base-1.0
COPY handler.py /app/handler.py
CMD ["python", "/app/handler.py"]
# Builds in ~30 seconds instead of 12 minutes
```

### Consider RunPod GitHub Integration
Instead of building locally and pushing to ghcr.io:
- Link the GitHub repo in RunPod dashboard
- Tag releases → RunPod builds on their infrastructure
- No image pull delay (already local to RunPod)
- Easy rollback from Builds tab

### Consider RunPod Cached Models
If ACE-Step weights are on HuggingFace (they are), RunPod can cache them automatically:
- Decouple model from Docker image → smaller image → faster cold starts
- You're not billed during model download
- Models cached at `/runpod-volume/huggingface-cache/hub/`

### Consider Modal as Alternative
Modal has an [official ACE-Step example](https://modal.com/docs/examples/generate_music):
- 2-4 second cold starts
- No Docker needed (Python decorators)
- Pay-per-second billing
- If RunPod continues to be painful, this is a viable pivot

---

## API Reference (Quick Access)

```bash
# Correct endpoint ID (old one is dead)
ENDPOINT=kppjd45rdtvh73
API_KEY=<your-runpod-api-key>  # stored in env var RUNPOD_API_KEY
GRAPHQL=https://api.runpod.io/graphql?api_key=$API_KEY

# Health check
curl -H "Authorization: Bearer $API_KEY" "https://api.runpod.ai/v2/$ENDPOINT/health"

# Submit ping job
curl -X POST "https://api.runpod.ai/v2/$ENDPOINT/run" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"ping": true}}'

# Check job status
curl -H "Authorization: Bearer $API_KEY" "https://api.runpod.ai/v2/$ENDPOINT/status/<JOB_ID>"

# Purge queue
curl -X POST "https://api.runpod.ai/v2/$ENDPOINT/purge-queue" \
  -H "Authorization: Bearer $API_KEY"

# Kill all workers (set maxWorkers=0)
curl -X POST "$GRAPHQL" -H "Content-Type: application/json" \
  -d '{"query": "mutation { saveEndpoint(input: { id: \"kppjd45rdtvh73\", name: \"sophia\", templateId: \"g8bgahscs8\", gpuIds: \"ADA_24\", workersMin: 0, workersMax: 0, idleTimeout: 30, executionTimeoutMs: 600000, scalerType: \"QUEUE_DELAY\", scalerValue: 1 }) { id } }"}'

# Update template image
curl -X POST "$GRAPHQL" -H "Content-Type: application/json" \
  -d '{"query": "mutation { saveTemplate(input: { id: \"g8bgahscs8\", name: \"sophia\", imageName: \"ghcr.io/kyl-solutions/sophia-runpod:<TAG>\", containerDiskInGb: 40, volumeInGb: 0, dockerArgs: \"\", env: [{key: \"RUNPOD_INIT_TIMEOUT\", value: \"900\"}] }) { id } }"}'

# Re-enable workers
# (same as kill but set workersMax: 1 or 2)
```

---

## Dead Endpoint Warning
- **CORRECT endpoint:** `kppjd45rdtvh73`
- **DEAD/OLD endpoint:** `gvghks6lzf4rfc` — DO NOT USE
- This changed when the endpoint settings were edited via dashboard. RunPod recreated it with a new ID.

## RunPod Balance
- Started: ~$9.96
- Current: ~$8-9 (burned on stuck workers and init loops)
- Template disk: 40GB
- GPU: ADA_24 (RTX 4090)

---

## Files Reference
- `handler.py` — v7.1 (module-level start, ping mode, lazy model load with error details)
- `Dockerfile` — 14.8GB image, CUDA 12.8, torch 2.10.0, ACE-Step 1.5, model weights baked in
- `.github/workflows/build-push.yml` — GitHub Actions CI, pushes to ghcr.io

## Next Session Checklist
- [ ] Set `RUNPOD_INIT_TIMEOUT=900` via GraphQL template update
- [ ] Pin `runpod==1.7.10` in Dockerfile
- [ ] Verify ghcr.io image visibility is PUBLIC
- [ ] Move model loading before `runpod.serverless.start()` (v8 handler)
- [ ] Build and deploy
- [ ] Test with ping (should work fast once init timeout is fixed)
- [ ] Test with real generation job
- [ ] If model load fails: check error details, may need CUDA/torch version fix
- [ ] Consider two-Dockerfile strategy for faster iteration
- [ ] Consider RunPod GitHub Integration as alternative to ghcr.io
