import os
import shutil


def cleanup_output() -> None:
    if os.path.exists("output") and os.path.isdir("output"):
        shutil.rmtree("output")
    os.makedirs("output", exist_ok=True)


def model_path(suffix: str, subfolder: str | None = None) -> str:
    subfolder_path = f"{subfolder}/" if subfolder else ""
    return f"output/{subfolder_path}output_{suffix}.bin"
