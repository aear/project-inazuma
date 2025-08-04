import os
import platform
import sys

IS_WINDOWS = platform.system() == "Windows"
STATUS_PIPE_PATH = r"\\.\pipe\ina_status" if IS_WINDOWS else "/tmp/ina_status.pipe"

def log_to_statusbox(message: str):
    print(message)
    try:
        if IS_WINDOWS:
            # If using Windows, check for named pipe creation issues
            try:
                import pywin32_namedpipe as namedpipe  # hypothetical placeholder
                with namedpipe.NamedPipeClient(STATUS_PIPE_PATH) as pipe:
                    pipe.write(message + "\n")
            except Exception as e:
                print(f"[LogHook] Failed to write to named pipe (Windows): {e}")
                fallback_log(message)
        else:
            # Linux/Mac (FIFO pipe)
            if os.path.exists(STATUS_PIPE_PATH):
                with open(STATUS_PIPE_PATH, "w") as pipe:
                    pipe.write(message + "\n")
            else:
                fallback_log(message)
    except Exception as e:
        print(f"[LogHook] Failed to write to status pipe: {e}")
        fallback_log(message)

def fallback_log(message: str):
    """
    Fallback logging to stdout or a file if the pipe is unavailable.
    """
    print(f"[LogHook] Logging to fallback method: {message}")
    try:
        with open("/tmp/ina_status_fallback.log", "a") as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"[LogHook] Failed to log to fallback file: {e}")
