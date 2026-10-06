from datetime import datetime

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .schemas import HealthResponse


app = FastAPI(
    title="Sales Forecasting API",
    description="Production-style API for the sales forecasting system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """API health check."""

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=["prophet"],
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8001")),
        reload=True,
    )
