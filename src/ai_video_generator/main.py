import uvicorn


def run() -> None:
    uvicorn.run(
        "ai_video_generator.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run()
