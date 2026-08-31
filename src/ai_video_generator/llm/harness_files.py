from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent / "resources" / "harnesses"


def load_harness(name: str) -> str:
    path = (HARNESS_ROOT / name).resolve()
    if HARNESS_ROOT.resolve() not in path.parents:
        raise ValueError("harness path escapes the harness resource directory")
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Harness 文件无法读取：{path}") from exc
    if not content:
        raise RuntimeError(f"Harness 文件为空：{path}")
    return content
