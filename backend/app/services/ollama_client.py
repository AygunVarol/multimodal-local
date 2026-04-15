
import os
import re
import requests
from typing import Any, Dict, List, Optional

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODELS = ["ministral-3:3b", "qwen3-vl:4b", "gemma3:4b-it-qat", "granite3.2-vision:latest"]

def _split_models(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[,\n]+", raw)
    models: List[str] = []
    seen = set()
    for part in parts:
        name = part.strip()
        if name and name not in seen:
            models.append(name)
            seen.add(name)
    return models

def _resolve_models() -> List[str]:
    models = _split_models(os.getenv("OLLAMA_MODELS"))
    if not models:
        models = list(DEFAULT_MODELS)
    default_env = os.getenv("OLLAMA_MODEL")
    if default_env:
        if default_env in models:
            models = [default_env] + [m for m in models if m != default_env]
        else:
            models = [default_env] + models
    return models

OLLAMA_MODELS = _resolve_models()
OLLAMA_MODEL = OLLAMA_MODELS[0] if OLLAMA_MODELS else DEFAULT_MODELS[0]

def available_models() -> List[str]:
    return list(OLLAMA_MODELS)

def default_model() -> str:
    return OLLAMA_MODEL

def resolve_model(requested: Optional[str]) -> str:
    if requested:
        candidate = requested.strip()
        if candidate:
            return candidate
    return default_model()

def _chat_payload(
    messages: List[Dict[str, Any]],
    images: Optional[list[str]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    if images and messages and messages[-1].get("role") == "user":
        messages[-1]["images"] = images
    return {"model": resolve_model(model), "messages": messages, "stream": False}

def chat(
    messages: List[Dict[str, Any]],
    images: Optional[list[str]] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = _chat_payload(messages, images, model=model)
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    content = None
    if isinstance(data, dict):
        if "message" in data and isinstance(data["message"], dict):
            content = data["message"].get("content")
        elif "response" in data:
            content = data.get("response")
    return {"content": content, "raw": data, "model": payload.get("model")}

def generate(prompt: str, images: Optional[list[str]] = None, model: Optional[str] = None) -> Dict[str, Any]:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {"model": resolve_model(model), "prompt": prompt, "stream": False}
    if images:
        payload["images"] = images
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return {"content": data.get("response"), "raw": data, "model": payload.get("model")}
