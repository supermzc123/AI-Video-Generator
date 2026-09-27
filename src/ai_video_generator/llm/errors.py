class LLMClientError(RuntimeError):
    def __init__(self, message: str, *, response_started: bool = False) -> None:
        super().__init__(message)
        self.response_started = response_started
