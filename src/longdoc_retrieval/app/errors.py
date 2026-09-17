class UserError(Exception):
    """Expected CLI failure. Mapped to a non-1 exit code by the CLI layer."""

    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code
