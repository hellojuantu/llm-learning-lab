from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COURSE = ROOT / "course"
PERSONAL_COURSE = ROOT / "personal" / "course"

EXCLUDED_DIRS = {".ipynb_checkpoints", "__pycache__"}
EXCLUDED_SUFFIXES = {".pyc"}
EXCLUDED_NAMES = {
    "classification_boundary.svg",
    "class_07_boundary.svg",
    "mini_project_points.svg",
}


def should_skip(path: Path) -> bool:
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return True
    if path.suffix in EXCLUDED_SUFFIXES:
        return True
    return path.name in EXCLUDED_NAMES


def iter_files(source: Path):
    for path in sorted(source.rglob("*")):
        if path.is_file() and not should_skip(path.relative_to(source)):
            yield path


def sync_tree(source: Path, target: Path, *, force: bool, dry_run: bool) -> tuple[int, int]:
    copied = 0
    skipped = 0
    for src in iter_files(source):
        rel = src.relative_to(source)
        dst = target / rel
        if dst.exists() and not force:
            skipped += 1
            continue
        copied += 1
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return copied, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 course 课程模板复制到 personal/course 本地工作区。"
    )
    parser.add_argument(
        "--lesson",
        help="只同步某一节，例如 08_pytorch_bridge。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖 personal/course 里已有文件。只在想重置本地作业时使用。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示会复制/跳过多少文件，不写入。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.lesson:
        source = COURSE / args.lesson
        target = PERSONAL_COURSE / args.lesson
        if not source.is_dir():
            raise SystemExit(f"Unknown lesson: {args.lesson}")
    else:
        source = COURSE
        target = PERSONAL_COURSE

    copied, skipped = sync_tree(source, target, force=args.force, dry_run=args.dry_run)
    action = "将复制" if args.dry_run else "已复制"
    print(f"{action}: {copied} 个文件")
    print(f"已存在，跳过: {skipped} 个文件")
    print(f"目标目录: {target}")
    if skipped and not args.force:
        print("已有个人文件已保留。只有想重置本地作业时才使用 --force。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
