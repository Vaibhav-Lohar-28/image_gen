# runpod-worker-comfy — Krea 2 Identity Edit / Head Swap edition

> [ComfyUI](https://github.com/comfyanonymous/ComfyUI) as a serverless API on [RunPod](https://www.runpod.io/), fully customized for **Lonecat's Krea2 Identity Edit & Head Swap workflow V6.0.2** (`lonecatsKrea2Identity_v602HeadSwapRMBG.json`, [Civitai](https://civitai.com/models/2803688/lonecats-krea2-identity-edit)).

This is a customized fork of [blib-la/runpod-worker-comfy](https://github.com/blib-la/runpod-worker-comfy). The Docker image bakes in every model, LoRA, upscaler, RMBG weight and custom node that this workflow needs, and `test_input.json` contains the workflow converted to ComfyUI's **API format** so it can run headless on RunPod.

---

## What the workflow does

Two photos in — one edited photo out:

- **Image 1 (`source_image.png`)** = the photo to keep (body, clothes, background, pose)
- **Image 2 (`reference_image.png`)** = the headshot whose face & hair get placed onto image 1

Pipeline (mirrors the V6.0.2 graph exactly):

1. **RMBG-2.0** removes the background of the headshot for better consistency (source photo stays raw, as in the original graph)
2. **AspectRatioSimplifier** sizes the output to the source image (long side clamped to 1640, divisible by 8)
3. **Animosity Krea2** checkpoint (int8_convrot) + **Krea 2 Identity Edit LoRA v1.2** via the [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) nodes (`Krea2EditModelPatch` + `Krea2EditGroundedEncode`, ref_boost 4 / ref_boost_a 0, `fit`)
4. **KSampler** — 8 steps, CFG 1, euler, simple (Krea 2 Turbo-style settings)
5. **Ultimate SD Upscale** 1.5× with the `1x-ITF-SkinDiffDetail-Lite` upscaler (4 steps, denoise 0.15)
6. **Post-production:** LC Clarity (Portrait) → LC Skin Beauty (Natural) → LC Image Adjust → LC Apply LUT (`LC_Crushed_Blacks.cube` @ 0.23)
7. **Image Saver Simple** saves the final PNG with Civitai-compatible metadata (via `Image Saver Metadata` + `CivitaiResourcesToHashMetadata`), into `output/Krea2 ID Edit/Lonecats/<date>/`

The head-swap instruction used (editable in `test_input.json`, node `"244"`):

> "Image 1 = the photo to keep. Image 2 = the face and hair to use. Put the person from image 2 onto the body in image 1. … Do not change image 1's clothes or background. Do not mix the two faces."

## What's inside the image

### Models (baked in at build time)

| File | ComfyUI folder | Source |
| ---- | -------------- | ------ |
| `Animosity_Krea2_V1.0_int8_convrot.safetensors` | `models/diffusion_models/Krea 2/Your models/` | [Civitai — Animosity (lonecatone23)](https://civitai.com/models/2596298/animosity), version *Krea2_V1.0_int8_convrot* |
| `qwen3vl_4b_fp8_scaled.safetensors` | `models/text_encoders/` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) |
| `krea2RealVae_v10.safetensors` | `models/vae/` | [martineux/altkreas](https://huggingface.co/martineux/altkreas) |
| `Krea 2/Utilities/krea2_identity_edit_v1_2.safetensors` | `models/loras/` | [conradlocke/krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit) |
| `Krea 2/Realism helpers/lenovo_krea2.safetensors` | `models/loras/` | [Civitai — Lenovo UltraReal, Krea 2 version](https://civitai.com/models/1662740/lenovo-ultrareal?modelVersionId=3075606) (ships switched **off** in the workflow) |
| `1x-ITF-SkinDiffDetail-Lite-v1.pth` | `models/upscale_models/` | [OpenModelDB](https://openmodeldb.info/models/1x-ITF-SkinDiffDetail-Lite-v1) |
| RMBG-2.0 (config, weights, code) | `models/RMBG/RMBG-2.0/` | [1038lab/RMBG-2.0](https://huggingface.co/1038lab/RMBG-2.0) (public mirror of briaai/RMBG-2.0) |
| `LC_Crushed_Blacks.cube` (+ other LUTs) | `models/luts/` | bundled with the LC123 node pack (auto-installed) |

### Custom nodes (installed from `snapshot_krea2.json` during build)

| Node pack | Used for |
| --------- | -------- |
| [lbouaraba/comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) | `Krea2EditModelPatch`, `Krea2EditGroundedEncode` — the core identity-edit nodes |
| [lonecatone23/ComfyUI_LC123_nodes](https://github.com/lonecatone23/ComfyUI_LC123_nodes) | Lonecat's nodes: AspectRatioSimplifier, LCAnySwitch, LCGetImage, LCVRAMCacheClear, LCClarity, LCSkinBeauty, LCImageAdjust, LCApplyLUT, LCAdvancedFolder, LCPositive, LCJoinStrings |
| [rgthree/rgthree-comfy](https://github.com/rgthree/rgthree-comfy) | Power Lora Loader (holds the optional Lenovo LoRA) |
| [ssitu/ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) | the 1.5× hi-res pass |
| [1038lab/ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) | `RMBG` background removal |
| [alexopus/ComfyUI-Image-Saver](https://github.com/alexopus/ComfyUI-Image-Saver) | `Image Saver Simple` + `Image Saver Metadata` (Civitai metadata) |
| [PBandDev/comfyui-lora-tag-hash-metadata](https://github.com/PBandDev/comfyui-lora-tag-hash-metadata) | `CivitaiResourcesToHashMetadata` |
| [vslinx/ComfyUI-vslinx-nodes](https://github.com/vslinx/ComfyUI-vslinx-nodes) | `vsLinx_BooleanFlip`, `vsLinx_AppendLorasFromNodeToString` |

Two deliberate API-conversion choices (documented here for transparency):

- The UI-only helper `JoinStrings` (KJNodes) was replaced with the identical `LCJoinStrings` (LC123) so we don't drag the huge KJNodes pack into the image for one string-concat node.
- `WidgetToString` (which reads the UNet name from the UI graph) was replaced by the constant model name string in the metadata node.
- Pure frontend nodes (notes, labels, bypassers, preview/comparison strips, the stitched 3-panel `SaveImage`) are not part of the API graph; the single output returned by the worker is the Image Saver Simple result.

### ComfyUI

Version **v0.34.0** — Krea 2 requires native ComfyUI support (older versions such as 0.3.30 cannot load it).

## Building the image

All credential build-args are **optional and blank by default** — fill in what you need:

```bash
docker build --target final --platform linux/amd64 \
  --build-arg HUGGINGFACE_ACCESS_TOKEN=<your-hf-token>     \ # blank ok: default mirrors are public
  --build-arg CIVITAI_API_TOKEN=<your-civitai-api-token>   \ # blank ok
  --build-arg ANIMOSITY_URL=<civitai-download-url>         \ # see below
  -t <your_dockerhub_username>/runpod-worker-comfy:krea2-identity .
```

### The Animosity Krea2 checkpoint

Civitai checkpoint URLs contain a per-version id, so paste the exact link for the **Krea2_V1.0_int8_convrot** version from [the Animosity page](https://civitai.com/models/2596298/animosity) (right-click its Download button → copy link, format `https://civitai.com/api/download/models/<versionId>`) and pass it as `ANIMOSITY_URL`.

If you already have the file locally, you can skip the URL entirely: drop it into

```
downloads/diffusion_models/Krea 2/Your models/Animosity_Krea2_V1.0_int8_convrot.safetensors
```

before building — everything under `downloads/` is merged into `/comfyui/models/` (Dockerfile stage 3).

> [!NOTE]
> Always add `--platform linux/amd64` — RunPod workers run on amd64. The full image is ~30 GB; use a Container Disk of 35 GB+ on RunPod.

## Config (runtime environment variables)

| Environment Variable        | Description                                                                          | Default          |
| --------------------------- | ------------------------------------------------------------------------------------ | ---------------- |
| `REFRESH_WORKER`            | Stop the worker after each finished job for a clean state.                           | `false`          |
| `COMFY_HOST`                | Host where ComfyUI is running.                                                       | `127.0.0.1:8188` |
| `COMFY_OUTPUT_PATH`         | Directory where ComfyUI stores generated images.                                     | `/comfyui/output`|
| `COMFY_POLLING_INTERVAL_MS` | Time between poll attempts (ms). Increase for longer workflows.                      | `250`            |
| `COMFY_POLLING_MAX_RETRIES` | Max poll attempts.                                                                     | `500`            |
| `COMFY_API_AVAILABLE_INTERVAL_MS` | Time between ComfyUI availability checks (ms).                                 | `50`             |
| `COMFY_API_AVAILABLE_MAX_RETRIES` | Max availability check attempts.                                               | `500`            |
| `SERVE_API_LOCALLY`         | Start the local API server for development (see [Local testing](#local-testing)).    | disabled         |

### Upload image to AWS S3 (optional)

Without S3 the final image is returned as a base64 string. To upload instead:

| Environment Variable       | Description                                             |
| -------------------------- | ------------------------------------------------------- |
| `BUCKET_ENDPOINT_URL`      | `https://<bucket>.s3.<region>.amazonaws.com`            |
| `BUCKET_ACCESS_KEY_ID`     | AWS access key ID                                       |
| `BUCKET_SECRET_ACCESS_KEY` | AWS secret access key                                   |

## Deploy on RunPod

1. Push the built image to Docker Hub (or any registry).
2. [Create a template](https://runpod.io/console/serverless/user/templates): Container Image `<your_username>/runpod-worker-comfy:krea2-identity`, Container Disk **35 GB**, your registry credentials.
3. [Create an endpoint](https://www.runpod.io/console/serverless/user/endpoints): GPU with **≥ 24 GB VRAM** (RTX 4090 / A5000 minimum, A100/L40S comfortable), Flash Boot enabled.

## Using the API

### JSON Request Body

```json
{
  "input": {
    "workflow": { ... see test_input.json ... },
    "images": [
      { "name": "source_image.png",    "image": "<base64 of the photo to keep>" },
      { "name": "reference_image.png", "image": "<base64 of the headshot>" }
    ]
  }
}
```

🚨 RunPod limits the request body to 10 MB (`/run`) / 20 MB (`/runsync`) — resize inputs to ~1–1.5 MP before base64-encoding. The workflow reads any aspect ratio; the output follows the source image's dimensions (long side ≤ 1640).

### Example (sync)

```bash
curl -X POST -H "Authorization: Bearer <api_key>" -H "Content-Type: application/json" \
  -d @test_input.json \
  https://api.runpod.ai/v2/<endpoint_id>/runsync
```

Response (base64 image, no S3 configured):

```json
{
  "output": { "message": "base64encodedimage", "status": "success" },
  "status": "COMPLETED"
}
```

### Useful tweaks inside `input.workflow`

| You want to… | Edit |
| ------------ | ---- |
| Change the instruction | node `"244"` → `inputs.positive` |
| Change the seed | nodes `"167"` and `"58"` → `inputs.seed` |
| Enable the optional Lenovo realism LoRA | node `"208"` → `LORA_1.on = true` (and strength) |
| Stop removing the headshot background | remove node `"234"` and point node `"236"`'s `any_01` to `["174", 0]` |
| Also remove the source photo background | add an `RMBG` node for node `"1"` and point node `"235"`'s `any_01` to it |
| Change upscale factor | node `"58"` → `inputs.upscale_by` |

The original UI-format workflow (loadable in the ComfyUI frontend) is kept in [test_resources/workflows/](./test_resources/workflows/).

## Bring your own models / nodes

- **Network Volume:** populate `/workspace/models/...` on a [network volume](https://docs.runpod.io/pods/storage/create-network-volumes) — the layout is mapped via [extra_model_paths.yaml](./src/extra_model_paths.yaml) (`/runpod-volume` base path, incl. `diffusion_models`, `text_encoders`, `loras`, `upscale_models`, …).
- **Local files at build time:** put anything under `downloads/` (mirroring the `models/` layout).
- **More custom nodes:** export a [ComfyUI Manager snapshot](https://github.com/ltdrdata/ComfyUI-Manager) and place it next to `snapshot_krea2.json`; it is restored during the build.

## Local testing

```bash
python -m venv venv && source ./venv/bin/activate
pip install -r requirements.txt
python -m unittest discover        # handler unit tests
```

Run the full worker locally (needs an NVIDIA GPU):

```bash
docker build --target final --platform linux/amd64 -t runpod-worker-comfy:krea2-identity .
docker-compose up                  # SERVE_API_LOCALLY=true, ports 8000 (API) + 8188 (ComfyUI)
curl -X POST -d @test_input.json http://localhost:8000/runsync
```

## Credits

- Worker base: [blib-la/runpod-worker-comfy](https://github.com/blib-la/runpod-worker-comfy)
- Workflow: [Lonecat's Krea2 Identity Edit & Head Swap V6.0.2](https://civitai.com/models/2803688/lonecats-krea2-identity-edit) by [lonecatone23](https://civitai.com/user/lonecatone23)
- Identity-edit nodes + LoRA: [lbouaraba/comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit), [conradlocke/krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit)
- Models: [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2), [martineux/altkreas](https://huggingface.co/martineux/altkreas), [Animosity (Civitai)](https://civitai.com/models/2596298/animosity), [Lenovo UltraReal (Civitai)](https://civitai.com/models/1662740/lenovo-ultrareal), [OpenModelDB](https://openmodeldb.info/models/1x-ITF-SkinDiffDetail-Lite-v1), [1038lab/RMBG-2.0](https://huggingface.co/1038lab/RMBG-2.0)
