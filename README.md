# runpod-worker-comfy — Krea 2 Identity Edit edition

> [ComfyUI](https://github.com/comfyanonymous/ComfyUI) as a serverless API on [RunPod](https://www.runpod.io/), pre-configured for **Lonecat's Krea2 Identity Edit & Head Swap workflow** ([Civitai](https://civitai.com/models/2803688/lonecats-krea2-identity-edit), V6.0.2 "Head Swap & RMBG").

This is a customized fork of [blib-la/runpod-worker-comfy](https://github.com/blib-la/runpod-worker-comfy). Instead of the generic FLUX/SDXL images, this repo bakes in everything the Krea 2 Identity Edit workflow needs: the Krea 2 model, its Qwen3-VL text encoder, its VAE, the Identity Edit LoRA, and the required custom nodes.

---

## What is this workflow?

The [Krea 2](https://huggingface.co/Comfy-Org/Krea-2) model (a 12.9B FLUX-family MMDiT) plus the [Krea 2 Identity Edit LoRA](https://huggingface.co/conradlocke/krea2-identity-edit) and the [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) nodes give you **instruction-based, identity-preserving image editing**:

- **Identity edit:** "create a photo of this person at a night market" — same face, same outfit, new scene/lighting
- **Head swap:** put the head (including hair) of person B onto the body/scene of person A
- **Local edits:** recolor, add/remove/replace objects, attribute and outfit changes
- **Two-input edits:** image 1 = scene, image 2 = person (order matters!)

Typical settings: **Turbo model, 8–12 steps, CFG 1.0, sampler `euler`, scheduler `simple`, LoRA strength 1.0**.

## What's inside the image

### Models (baked in at build time)

| File | ComfyUI folder | Source |
| ---- | -------------- | ------ |
| `krea2_turbo_fp8_scaled.safetensors` | `models/diffusion_models/` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) |
| `qwen3vl_4b_fp8_scaled.safetensors` | `models/text_encoders/` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) |
| `qwen_image_vae.safetensors` | `models/vae/` | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) |
| `Krea2/krea2_identity_edit_v1_2.safetensors` | `models/loras/` | [conradlocke/krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit) |

Optional (uncomment in the `Dockerfile`): `krea2_raw_fp8_scaled.safetensors` for removal/deletion edits at CFG ~3.

### Custom nodes (installed from `snapshot_krea2.json`)

| Node pack | Provides |
| --------- | -------- |
| [lbouaraba/comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) | `Krea2EditModelPatch`, `Krea2EditGroundedEncode` — the core identity-edit nodes |
| [lonecatone23/ComfyUI_LC123_nodes](https://github.com/lonecatone23/ComfyUI_LC123_nodes) | Lonecat's utility / post-production nodes used by his workflows |

### ComfyUI

Version **v0.34.0** — the first requirement is native Krea 2 support in ComfyUI core, which older versions (e.g. 0.3.30) do not have.

## Hugging Face token

The model mirrors used by default (`Comfy-Org/Krea-2`) are public, so you can build **without** a token. A token **is** required if you switch the downloads to gated repositories such as the official [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo).

The token is wired through the `HUGGINGFACE_ACCESS_TOKEN` build-arg and used automatically by every download in the `Dockerfile`. Leave it blank if you don't need it:

```bash
# without a token (public mirrors)
docker build --target final --platform linux/amd64 \
  -t <your_dockerhub_username>/runpod-worker-comfy:krea2-identity .

# with a token (for gated repos)
docker build --target final --platform linux/amd64 \
  --build-arg HUGGINGFACE_ACCESS_TOKEN=<your-huggingface-token> \
  -t <your_dockerhub_username>/runpod-worker-comfy:krea2-identity .
```

> [!NOTE]
> Always add `--platform linux/amd64` — RunPod workers run on amd64.

## Config

| Environment Variable        | Description                                                                                                                                                                           | Default         |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `REFRESH_WORKER`            | Stop the worker after each finished job to have a clean state, see [official documentation](https://docs.runpod.io/docs/handler-additional-controls#refresh-worker).                  | `false`         |
| `COMFY_HOST`                | Host where ComfyUI is running.                                                                                                                                                        | `127.0.0.1:8188`|
| `COMFY_OUTPUT_PATH`         | Directory where ComfyUI stores generated images.                                                                                                                                      | `/comfyui/output`|
| `COMFY_POLLING_INTERVAL_MS` | Time to wait between poll attempts in milliseconds.                                                                                                                                   | `250`           |
| `COMFY_POLLING_MAX_RETRIES` | Maximum number of poll attempts. Increase this the longer your workflow runs.                                                                                                         | `500`           |
| `COMFY_API_AVAILABLE_INTERVAL_MS` | Time to wait between ComfyUI API availability checks in milliseconds.                                                                                                          | `50`            |
| `COMFY_API_AVAILABLE_MAX_RETRIES` | Maximum number of ComfyUI API availability check attempts.                                                                                                                       | `500`           |
| `SERVE_API_LOCALLY`         | Enable local API server for development and testing. See [Local Testing](#local-testing).                                                                                             | disabled        |

### Upload image to AWS S3

Only needed if you want the generated picture uploaded to AWS S3. Without it, the image is returned as a base64-encoded string.

| Environment Variable       | Description                                             | Example                                      |
| -------------------------- | ------------------------------------------------------- | -------------------------------------------- |
| `BUCKET_ENDPOINT_URL`      | The endpoint URL of your S3 bucket.                     | `https://<bucket>.s3.<region>.amazonaws.com` |
| `BUCKET_ACCESS_KEY_ID`     | Your AWS access key ID for accessing the S3 bucket.     | `AKIAIOSFODNN7EXAMPLE`                       |
| `BUCKET_SECRET_ACCESS_KEY` | Your AWS secret access key for accessing the S3 bucket. | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`   |

## Use the Docker image on RunPod

### Create your template (optional)

- Create a [new template](https://runpod.io/console/serverless/user/templates) by clicking on `New Template`
- In the dialog, configure:
  - Template Name: `runpod-worker-comfy-krea2`
  - Template Type: serverless
  - Container Image: `<your_dockerhub_username>/runpod-worker-comfy:krea2-identity`
  - Container Registry Credentials: your Docker Hub credentials (the image is private unless you push it publicly)
  - Container Disk: **35 GB** (the image with models is ~30 GB)
  - (optional) Environment Variables: [Configure S3](#upload-image-to-aws-s3)
- Click on `Save Template`

### Create your endpoint

- Navigate to [`Serverless > Endpoints`](https://www.runpod.io/console/serverless/user/endpoints) and click on `New Endpoint`
- In the dialog, configure:
  - Endpoint Name: `comfy-krea2`
  - Worker configuration: **a GPU with at least 24 GB VRAM** (e.g. RTX 4090 / A5000 for tight fp8 runs, A100/L40S comfortably)
  - Active Workers: `0` (whatever makes sense for you)
  - Max Workers: `3` (whatever makes sense for you)
  - GPUs/Worker: `1`
  - Idle Timeout: `5`
  - Flash Boot: `enabled`
  - Select Template: `runpod-worker-comfy-krea2`
- Click `deploy`

### GPU recommendations

| Model                | Minimum VRAM | Container Size |
| -------------------- | ------------ | -------------- |
| Krea 2 (fp8) + LoRA  | 24 GB        | ~30 GB         |

## API specification

The worker accepts the standard RunPod serverless input. Only the fields sent via `input` are described here; see the [official documentation](https://docs.runpod.io/docs/serverless-usage) for the full job format.

### JSON Request Body

```json
{
  "input": {
    "workflow": {},
    "images": [
      {
        "name": "source_image.png",
        "image": "base64_encoded_string"
      },
      {
        "name": "reference_image.png",
        "image": "base64_encoded_string"
      }
    ]
  }
}
```

### Fields

| Field Path       | Type   | Required | Description                                                                                                                               |
| ---------------- | ------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `input`          | Object | Yes      | The top-level object containing the request data.                                                                                          |
| `input.workflow` | Object | Yes      | The ComfyUI workflow, **exported in API format**.                                                                                          |
| `input.images`   | Array  | No       | Images uploaded into ComfyUI's `input` folder; the workflow references them by `name`. For this workflow: the scene to edit + the identity reference photo. |

#### "input.images"

Each image needs a unique `name` and a base64-encoded `image`.

🚨 The request body is limited to 10 MB for `/run` and 20 MB for `/runsync`, so don't send huge images (resize inputs to ~1–1.5 MP).

## Interact with your RunPod API

1. **Generate an API Key** in [User Settings > API Keys](https://www.runpod.io/console/serverless/user/settings).
2. **Find your [Endpoint ID](https://www.runpod.io/console/serverless)** (shown underneath the endpoint name).

### Health status

```bash
curl -H "Authorization: Bearer <api_key>" https://api.runpod.ai/v2/<endpoint_id>/health
```

### Identity edit / head swap example

The example below sends two images — the scene/person to edit (`source_image.png`) and the identity to apply (`reference_image.png`) — together with the workflow from [test_input.json](./test_input.json):

```bash
curl -X POST -H "Authorization: Bearer <api_key>" -H "Content-Type: application/json" \
  -d '{
    "input": {
      "workflow": { ... API-format workflow, see test_input.json ... },
      "images": [
        {"name": "source_image.png", "image": "<base64>"},
        {"name": "reference_image.png", "image": "<base64>"}
      ]
    }
  }' \
  https://api.runpod.ai/v2/<endpoint_id>/runsync
```

A head swap is the same graph with an instruction like:

> "Remove the head of the person in the scene. Put the head from the reference photo onto the person, keep the hair, its color and length, the eye color, the nose and the jaw."

Example response (base64 image, no S3 configured):

```json
{
  "delayTime": 2188,
  "executionTime": 22970,
  "id": "sync-c0cd1eb2-0699-4416-9333-5e624a80ec3c-e1",
  "output": { "message": "base64encodedimage", "status": "success" },
  "status": "COMPLETED"
}
```

## How to get the workflow from ComfyUI?

- Open ComfyUI in the browser
- Open the `Settings` (gear icon) and enable `Dev mode Options`
- Click `Save (API Format)` — this downloads `workflow_api.json`
- Put its content into the `workflow` field of your request

The UI-format version of the workflow lives in [test_resources/workflows/](./test_resources/workflows/).

## Bring Your Own Models and Nodes

### Network Volume

Using a Network Volume allows you to store and access custom models (e.g. extra LoRAs or the Krea 2 Raw variant):

1. [Create a Network Volume](https://docs.runpod.io/pods/storage/create-network-volumes).
2. Deploy a temporary GPU instance attached to the volume and populate it:

   ```bash
   cd /workspace
   for i in checkpoints clip clip_vision configs controlnet embeddings loras upscale_models vae unet diffusion_models text_encoders; do mkdir -p models/$i; done
   ```

3. [Terminate the temporary instance](https://docs.runpod.io/docs/pods#terminating-a-pod).
4. Attach the volume in your endpoint configuration under `Advanced > Select Network Volume`.

The folder layout is mapped automatically via [extra_model_paths.yaml](./src/extra_model_paths.yaml) (`/runpod-volume` base path).

### Custom Docker Image

To bake additional models into the image, add download commands to stage 2 of the `Dockerfile`:

```Dockerfile
RUN curl -fL -o models/loras/Krea2/my_extra_lora.safetensors https://huggingface.co/<user>/<repo>/resolve/main/my_extra_lora.safetensors
```

To add custom nodes, export a [ComfyUI Manager snapshot](https://github.com/ltdrdata/ComfyUI-Manager?tab=readme-ov-file#snapshot-manager), save it in the root of this repo (next to `snapshot_krea2.json`), and it will be restored during the build.

> [!NOTE]
> - Some custom nodes download additional models during installation, which increases image size
> - Many custom nodes increase ComfyUI's startup time

## Local testing

The tests use [test_input.json](./test_input.json).

### Setup

1. Python >= 3.10
2. `python -m venv venv` and activate it
3. `pip install -r requirements.txt`

### Testing the RunPod handler

- Run all tests: `python -m unittest discover`
- Run one test: `python -m unittest tests.test_rp_handler.TestRunpodWorkerComfy.test_valid_input_with_workflow_only`

### Local API

`SERVE_API_LOCALLY=true` starts an API server that simulates the RunPod worker environment (already set in `docker-compose.yml`):

```bash
docker-compose up
```

Then send jobs to `http://localhost:8000/runsync` with the same JSON shape as the RunPod API, e.g.:

```bash
curl -X POST -H "Content-Type: application/json" -d @test_input.json http://localhost:8000/runsync
```

ComfyUI itself is reachable at `http://localhost:8188`.

## Credits

- Worker base: [blib-la/runpod-worker-comfy](https://github.com/blib-la/runpod-worker-comfy)
- Workflow: [Lonecat's Krea2 Identity Edit & Head Swap](https://civitai.com/models/2803688/lonecats-krea2-identity-edit) by [lonecatone23](https://civitai.com/user/lonecatone23)
- Krea 2 nodes: [lbouaraba/comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit)
- Identity Edit LoRA: [conradlocke/krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit)
- Models: [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) (repackaged [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo))
