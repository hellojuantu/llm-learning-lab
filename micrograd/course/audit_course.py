from __future__ import annotations

import json
import os
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "course"

REQUIRED_LESSON_FILES = [
    "README.md",
    "preview.ipynb",
    "class.ipynb",
    "homework.ipynb",
    "review_prompt.md",
]

HOMEWORK_MARKERS = {
    "worked_example": ["完整例子", "example"],
    "modify": ["Modify", "改一行", "只改"],
    "todo": ["TODO"],
    "qa_check": ["qa_check"],
    "debug": ["Debug Lab", "Debug"],
    "hint": ["Show / Hide 提示"],
    "answer": ["Show / Hide 答案"],
    "checklist": ["提交清单"],
}

CLASS_MARKERS = {
    "predict": ["Predict"],
    "run": ["Run"],
    "modify": ["Modify"],
    "qa_check": ["qa_check"],
}


def read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source


def notebook_source(path: Path) -> str:
    nb = read_notebook(path)
    return "\n".join(cell_source(cell) for cell in nb.get("cells", []))


def course_notebooks() -> list[Path]:
    return sorted([*COURSE.glob("*.ipynb"), *COURSE.glob("*/*.ipynb")])


def lesson_dirs() -> list[Path]:
    return sorted(
        path
        for path in COURSE.iterdir()
        if path.is_dir() and path.name[:2].isdigit()
    )


def audit_structure() -> list[str]:
    errors: list[str] = []
    for lesson in lesson_dirs():
        for filename in REQUIRED_LESSON_FILES:
            if not (lesson / filename).exists():
                errors.append(f"{lesson.name}: missing {filename}")
    return errors


def audit_homework_shape() -> list[str]:
    errors: list[str] = []
    for lesson in lesson_dirs():
        path = lesson / "homework.ipynb"
        source = notebook_source(path)
        for name, markers in HOMEWORK_MARKERS.items():
            if not any(marker in source for marker in markers):
                errors.append(f"{lesson.name}/homework.ipynb: missing {name}")
    return errors


def audit_notebook_answer_hiding() -> list[str]:
    errors: list[str] = []
    forbidden = ["def qa_check_", "`assert ", "### qa_check"]
    for path in course_notebooks():
        source = notebook_source(path)
        for marker in forbidden:
            if marker in source:
                errors.append(f"{path.relative_to(ROOT)}: visible answer/check marker {marker!r}")
    return errors


def audit_per_exercise_support() -> list[str]:
    errors: list[str] = []
    for path in course_notebooks():
        nb = read_notebook(path)
        cells = nb.get("cells", [])
        for index, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            source = cell_source(cell)
            if "qa_check(" not in source:
                continue
            following = "\n".join(
                cell_source(next_cell)
                for next_cell in cells[index + 1:index + 4]
                if next_cell.get("cell_type") == "markdown"
            )
            if "Show / Hide 本题提示" not in following:
                errors.append(f"{path.relative_to(ROOT)}: cell {index + 1} missing per-exercise hint")
            if "Show / Hide 本题答案" not in following:
                errors.append(f"{path.relative_to(ROOT)}: cell {index + 1} missing per-exercise answer")
    return errors


def audit_class_shape() -> list[str]:
    errors: list[str] = []
    for lesson in lesson_dirs():
        path = lesson / "class.ipynb"
        source = notebook_source(path)
        for name, markers in CLASS_MARKERS.items():
            if not any(marker in source for marker in markers):
                errors.append(f"{lesson.name}/class.ipynb: missing {name}")
    return errors


def audit_notebook_json() -> list[str]:
    errors: list[str] = []
    for path in course_notebooks():
        try:
            nb = read_notebook(path)
        except Exception as exc:  # pragma: no cover - diagnostic script
            errors.append(f"{path}: invalid json: {exc}")
            continue
        if nb.get("nbformat") != 4:
            errors.append(f"{path}: nbformat is not 4")
        if not nb.get("cells"):
            errors.append(f"{path}: no cells")
    return errors


def audit_blank_execution() -> list[str]:
    """Execute course notebooks in blank TODO state.

    The point is not to pass learner tests; blank TODOs should print helpful
    messages instead of crashing Run All.
    """

    errors: list[str] = []
    old_cwd = Path.cwd()
    for path in course_notebooks():
        nb = read_notebook(path)
        namespace = {"__name__": "__main__"}
        try:
            os.chdir(path.parent)
            for index, cell in enumerate(nb.get("cells", []), start=1):
                if cell.get("cell_type") == "code":
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        exec(cell_source(cell), namespace)
        except Exception as exc:  # pragma: no cover - diagnostic script
            errors.append(f"{path}: cell {index}: {type(exc).__name__}: {exc}")
        finally:
            os.chdir(old_cwd)
    for generated in COURSE.glob("*/*.svg"):
        generated.unlink()
    return errors


def main() -> int:
    checks = {
        "structure": audit_structure(),
        "notebook_json": audit_notebook_json(),
        "answer_hiding": audit_notebook_answer_hiding(),
        "per_exercise_support": audit_per_exercise_support(),
        "class_shape": audit_class_shape(),
        "homework_shape": audit_homework_shape(),
        "blank_execution": audit_blank_execution(),
    }
    failed = False
    for name, errors in checks.items():
        if errors:
            failed = True
            print(f"FAIL {name}")
            for error in errors:
                print("  -", error)
        else:
            print(f"OK {name}")
    print(f"lesson_count={len(lesson_dirs())}")
    print(f"notebook_count={len(course_notebooks())}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
