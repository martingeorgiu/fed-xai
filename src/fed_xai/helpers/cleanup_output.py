import os
import shutil


def cleanup_output() -> None:
    if os.path.exists("output") and os.path.isdir("output"):
        shutil.rmtree("output")
    os.makedirs("output", exist_ok=True)


def model_path(name: str) -> str:
    return f"output/output_{name}.bin"
