"""
Main API Application Module

This module defines the main FastAPI application and configures all routes.

What Is This? (Explain Like I'm 5)
===============================
This is like the main control panel for our AI toy. Just like a TV remote
control that has buttons for different channels, our main API application
connects all the different parts of our AI system together so they can work
as one complete system.
"""

import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.api.endpoints import router
from src.utils.logger import get_logger

# Get logger instance
logger = get_logger("main_api")

# Create FastAPI app
app = FastAPI(
    title="LLM Text Classification API",
    description="API for text classification using Large Language Models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router, prefix="/api/v1")

# Add root endpoint
@app.get("/")
async def root():
    """
    Root endpoint.
    
    Returns a welcome message and API information.
    """
    logger.info("Root endpoint accessed")
    
    return {
        "message": "Welcome to the LLM Text Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Add startup event
@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    
    This function is called when the application starts up.
    """
    logger.info("Starting LLM Text Classification API")
    logger.info("API is ready to serve requests")


# Add shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler.
    
    This function is called when the application shuts down.
    """
    logger.info("Shutting down LLM Text Classification API")


if __name__ == "__main__":
    # Run the application
    logger.info("Starting API server")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )