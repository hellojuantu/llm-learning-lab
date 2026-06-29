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

CLASS_MIN_CELLS = 18
CLASS_MIN_CHARS = 3500


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


def audit_qa_instruction_comments() -> list[str]:
    errors: list[str] = []
    for path in course_notebooks():
        nb = read_notebook(path)
        for index, cell in enumerate(nb.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            source = cell_source(cell)
            if "qa_check(" in source and "# 填写说明：" not in source:
                errors.append(f"{path.relative_to(ROOT)}: cell {index} missing fill instructions")
    return errors


def audit_homework_ladder_order() -> list[str]:
    errors: list[str] = []
    for lesson in lesson_dirs():
        path = lesson / "homework.ipynb"
        nb = read_notebook(path)
        first_example = None
        first_modify = None
        for index, cell in enumerate(nb.get("cells", []), start=1):
            if cell.get("cell_type") != "markdown":
                continue
            first_line = (cell_source(cell).splitlines() or [""])[0]
            if first_example is None and first_line.startswith("## 完整例子"):
                first_example = index
            if first_modify is None and first_line.startswith("## Modify"):
                first_modify = index
        if first_example is None:
            errors.append(f"{path.relative_to(ROOT)}: missing worked example section")
        if first_modify is not None and first_example is not None and first_modify < first_example:
            errors.append(f"{path.relative_to(ROOT)}: Modify appears before worked example")
    return errors


def audit_contextual_hints() -> list[str]:
    errors: list[str] = []
    generic_fragments = [
        "先参考上面的完整例子，只改 TODO 处",
        "测试失败时，先判断错在数学、Python、计算图还是训练循环",
    ]
    for path in course_notebooks():
        source = notebook_source(path)
        for fragment in generic_fragments:
            if fragment in source:
                errors.append(f"{path.relative_to(ROOT)}: generic hint remains")
    return errors


def audit_visual_feedback() -> list[str]:
    errors: list[str] = []
    for path in course_notebooks():
        nb = read_notebook(path)
        for index, cell in enumerate(nb.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            source = cell_source(cell)
            writes_svg = ".svg" in source and "write_text" in source
            shows_svg = "show_svg(" in source or "display(SVG" in source
            if writes_svg and not shows_svg:
                errors.append(f"{path.relative_to(ROOT)}: cell {index} writes svg without displaying it")
    return errors


REVIEW_REQUIRED_KEYWORDS = {
    "00_math_bootcamp": ["导数", "偏导", "梯度", "链式法则", "ReLU"],
    "01_gradient_homework": ["局部导数", "多路径", "L=a*a", "ReLU"],
    "02_value_usage": ["data", "grad", "backward", "重复"],
    "03_value_source_reading": ["self", "other", "out", "_backward", "拓扑"],
    "04_tinyvalue_homework": ["__add__", "__mul__", "__pow__", "relu", "build_topo"],
    "05_neuron_mlp": ["Neuron", "Layer", "MLP", "参数", "bias"],
    "06_training_loop": ["forward", "loss", "zero_grad", "backward", "update"],
    "07_classification_boundary": ["score", "y * score", "margin", "hinge loss", "边界"],
    "08_pytorch_bridge": ["Tensor", "requires_grad", "zero_grad", "optimizer.step", "shape"],
    "09_gradient_debugging": ["loss", "grad", "learning_rate", "dead ReLU", "checklist"],
    "10_mini_project": ["数据", "MLP(2,[4,1])", "accuracy", "decision boundary", "答辩"],
    "11_structure_next_route": ["engine.py", "nn.py", "PyTorch", "D2L", "nanoGPT"],
}


def audit_review_prompts() -> list[str]:
    errors: list[str] = []
    generic_fragments = [
        "请重点检查：",
        "我是否真的理解了每个作业台阶",
        "我是否理解 Debug Lab 的错误原因",
        "我是否能把本节接到 micrograd 主线",
    ]
    for lesson in lesson_dirs():
        path = lesson / "review_prompt.md"
        text = path.read_text(encoding="utf-8")
        for fragment in generic_fragments:
            if fragment in text:
                errors.append(f"{path.relative_to(ROOT)}: generic review fragment remains")
        if "请按这个顺序检查我：" not in text:
            errors.append(f"{path.relative_to(ROOT)}: missing ordered review questions")
        question_count = sum(1 for line in text.splitlines() if line[:1].isdigit() and ". " in line)
        if question_count < 7:
            errors.append(f"{path.relative_to(ROOT)}: too few review questions ({question_count})")
        for keyword in REVIEW_REQUIRED_KEYWORDS.get(lesson.name, []):
            if keyword not in text:
                errors.append(f"{path.relative_to(ROOT)}: missing lesson keyword {keyword!r}")
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


def audit_class_depth() -> list[str]:
    errors: list[str] = []
    for lesson in lesson_dirs():
        path = lesson / "class.ipynb"
        nb = read_notebook(path)
        source = notebook_source(path)
        if len(nb.get("cells", [])) < CLASS_MIN_CELLS:
            errors.append(
                f"{lesson.name}/class.ipynb: too few cells ({len(nb.get('cells', []))})"
            )
        if len(source) < CLASS_MIN_CHARS:
            errors.append(f"{lesson.name}/class.ipynb: too short ({len(source)} chars)")
        if "What To Remember" not in source:
            errors.append(f"{lesson.name}/class.ipynb: missing What To Remember")
        if "课堂检查" not in source:
            errors.append(f"{lesson.name}/class.ipynb: missing classroom check section")
    return errors


def audit_no_checkpoints() -> list[str]:
    errors: list[str] = []
    for checkpoint in COURSE.rglob(".ipynb_checkpoints"):
        errors.append(f"{checkpoint.relative_to(ROOT)} should not be committed")
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
    old_backend = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
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
    if old_backend is None:
        os.environ.pop("MPLBACKEND", None)
    else:
        os.environ["MPLBACKEND"] = old_backend
    for generated in COURSE.glob("*/*.svg"):
        generated.unlink()
    for pycache in COURSE.rglob("__pycache__"):
        for child in pycache.iterdir():
            child.unlink()
        pycache.rmdir()
    return errors


def main() -> int:
    checks = {
        "structure": audit_structure(),
        "notebook_json": audit_notebook_json(),
        "answer_hiding": audit_notebook_answer_hiding(),
        "per_exercise_support": audit_per_exercise_support(),
        "qa_instruction_comments": audit_qa_instruction_comments(),
        "homework_ladder_order": audit_homework_ladder_order(),
        "contextual_hints": audit_contextual_hints(),
        "visual_feedback": audit_visual_feedback(),
        "review_prompts": audit_review_prompts(),
        "class_shape": audit_class_shape(),
        "class_depth": audit_class_depth(),
        "homework_shape": audit_homework_shape(),
        "no_checkpoints": audit_no_checkpoints(),
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
