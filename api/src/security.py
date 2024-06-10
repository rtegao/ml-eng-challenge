from fastapi.security import APIKeyHeader
from fastapi import FastAPI, Depends, HTTPException
from config import settings


# Define a dependency to check the API key in the header
api_key_header = APIKeyHeader(name="X-API-Key")

# Dependency function to validate API key
async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key