from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.documents.router import router as documents_router
from app.api.sync.router import router as sync_router
from app.config.logger import app_logger, log_request_end, log_request_error, log_request_start
from app.config.settings import settings


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing information using Loguru."""
    start_time = datetime.now()
    
    # Log request start
    log_request_start(request)
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = (datetime.now() - start_time).total_seconds()
        
        # Log response
        log_request_end(request, response.status_code, process_time)
        
        return response
        
    except Exception as e:
        # Log error
        process_time = (datetime.now() - start_time).total_seconds()
        log_request_error(request, e, process_time)
        raise


@app.get("/", tags=["health"])
async def root():
    """Root endpoint with basic API information."""
    app_logger.info("Root endpoint accessed")
    return {
        "message": "Welcome to the GCE English backend",
        "version": settings.app_version,
        "status": "operational",
        "docs": "/docs",
        "redoc": "/redoc",
    }


app.include_router(documents_router)
app.include_router(sync_router)



if __name__ == "__main__":
    import uvicorn
    app_logger.info("Starting GCE English backend server")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
