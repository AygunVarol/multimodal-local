
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.chat import router as chat_router
from .routes.dataqa import router as dataqa_router
from .routes.kg import router as kg_router
from .routes.display import router as display_router
from .services.ollama_client import available_models, default_model

app = FastAPI(title="Local Smart-Space Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "model": default_model(), "models": available_models()}

app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(dataqa_router, prefix="/api", tags=["data"])
app.include_router(kg_router, prefix="/api", tags=["kg"])
app.include_router(display_router, prefix="/api", tags=["display"])
