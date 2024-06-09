import uvicorn
from fastapi import FastAPI
from src.routes import router
from fastapi.openapi.utils import get_openapi
from swagger import custom_openapi
from config import settings

app = FastAPI()

app.include_router(router=router)


# Enable automatic Swagger UI generation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# URL for Swagger UI
app.openapi_url = settings.SWAGGER_UI

# Optionally, you can set the description for your API
app.description = "Your API Description"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
