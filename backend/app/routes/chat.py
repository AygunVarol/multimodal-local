
from fastapi import APIRouter, File, Form, Header, UploadFile
from typing import List, Optional
from ..services.ollama_client import chat
from ..utils.images import files_to_base64
from ..services.kg_store import KGStore

router = APIRouter()
kg = KGStore()

def _header_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}

@router.post("/chat")
async def chat_endpoint(
    message: str = Form(...),
    images: Optional[List[UploadFile]] = File(None),
    model: Optional[str] = Form(None),
    kg_disabled: bool = Form(False),
    x_kg_disabled: Optional[str] = Header(None, alias="X-KG-Disabled"),
):
    images_b64 = await files_to_base64(images) if images else None
    use_kg = not (kg_disabled or _header_truthy(x_kg_disabled))
    system_content = "You are a helpful local assistant. If images are provided, analyze them. Keep responses concise unless details are requested."
    if use_kg:
        system_content += "\n\nRelevant KG facts (if any):\n" + kg.short_context_for(message)
    system = {
        "role": "system",
        "content": system_content,
    }
    user = {"role": "user", "content": message}
    try:
        result = chat([system, user], images=images_b64, model=model)
        return {
            "content": result.get("content") or "No content returned.",
            "raw": result.get("raw"),
            "model": result.get("model"),
        }
    except Exception as e:
        if images_b64:
            return {"content":"The configured model might not support images. Try a vision-capable model (e.g., 'llava').","error":str(e)}
        return {"content":"Error contacting the local model.","error":str(e)}
