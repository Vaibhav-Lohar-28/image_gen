variable "DOCKERHUB_REPO" {
  default = "timpietruskyblibla"
}

variable "DOCKERHUB_IMG" {
  default = "runpod-worker-comfy"
}

variable "RELEASE_VERSION" {
  default = "latest"
}

variable "HUGGINGFACE_ACCESS_TOKEN" {
  default = ""
}

group "default" {
  targets = ["base", "krea2-identity"]
}

target "base" {
  context = "."
  dockerfile = "Dockerfile"
  target = "base"
  platforms = ["linux/amd64"]
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}-base"]
}

target "krea2-identity" {
  context = "."
  dockerfile = "Dockerfile"
  target = "final"
  args = {
    HUGGINGFACE_ACCESS_TOKEN = "${HUGGINGFACE_ACCESS_TOKEN}"
  }
  platforms = ["linux/amd64"]
  tags = ["${DOCKERHUB_REPO}/${DOCKERHUB_IMG}:${RELEASE_VERSION}-krea2-identity"]
  inherits = ["base"]
}
