import uvicorn

from ai_video_generator.api import app
from ai_video_generator.config import get_settings


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
