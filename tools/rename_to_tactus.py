#!/usr/bin/env python3
"""Rename the analysis library from OsuTest.BeatAnalysis to Tactus.

The library is published as its own repository, and OsuTest is the name of the application that happened to grow it
rather than a name for a library. The package identifier keeps a .NET suffix while the assembly and namespace do not:
`Tactus.NET` is what someone searches for and what the repository is called, and `Tactus` is what they type in code,
so that a type reads `Tactus.BeatGrid` rather than `Tactus.NET.BeatGrid`. The two differing is ordinary - the
`ppy.osu.Framework` package this library depends on ships an assembly called `osu.Framework.dll`.

Deliberately not rewritten:

* `tools/` - the scripts there record refactors that actually happened. Rewriting the names in them would falsify
  that record, and none of them run as part of the build.
* `bin/` and `obj/` - build output, deleted instead so nothing stale survives the rename.
* `all.txt` and `r1.txt` .. `r5.txt` - captured build logs at the repository root.
"""

import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = "OsuTest.BeatAnalysis"
NEW = "Tactus"
PACKAGE = "Tactus.NET"
REPOSITORY = "https://github.com/CattyCathy/Tactus.NET"

OLD_DIR = os.path.join(ROOT, "OsuTest", OLD)
NEW_DIR = os.path.join(ROOT, "OsuTest", NEW)

EXTENSIONS = {".cs", ".csproj", ".sln", ".slnf", ".md", ".txt", ".props", ".targets"}

SKIP_DIRS = {"bin", "obj", ".git", "headless", "venv", "node_modules"}
SKIP_RELATIVE = {os.path.join("tools", ""), os.path.join("headless", "")}
SKIP_FILES = {"all.txt", "r1.txt", "r2.txt", "r3.txt", "r4.txt", "r5.txt"}


def skipped(path):
    relative = os.path.relpath(path, ROOT)

    if os.path.basename(path) in SKIP_FILES and os.path.dirname(relative) == ".":
        return True

    return any(relative.startswith(prefix) for prefix in SKIP_RELATIVE)


def walk_text_files():
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for name in files:
            path = os.path.join(base, name)

            if os.path.splitext(name)[1].lower() not in EXTENSIONS:
                continue

            if skipped(path):
                continue

            yield path


def rename_directory():
    if not os.path.isdir(OLD_DIR):
        print(f"nothing to rename: {OLD_DIR} does not exist")
        return

    if os.path.isdir(NEW_DIR):
        sys.exit(f"refusing to overwrite the existing {NEW_DIR}")

    os.rename(OLD_DIR, NEW_DIR)
    print(f"renamed {os.path.relpath(OLD_DIR, ROOT)} -> {os.path.relpath(NEW_DIR, ROOT)}")


def rename_project_file():
    old = os.path.join(NEW_DIR, OLD + ".csproj")
    new = os.path.join(NEW_DIR, NEW + ".csproj")

    if os.path.isfile(old):
        os.rename(old, new)
        print(f"renamed {os.path.basename(old)} -> {os.path.basename(new)}")


def read_text(path):
    """Read a file, reporting the encoding it was stored in so it can be written back the same way.

    Visual Studio writes its solution and project files in whatever encoding it feels like - the solution here is
    UTF-16 with a byte order mark - and decoding those as UTF-8 fails on the first byte. Rewriting a file in a
    different encoding than it arrived in would show up as a whole-file diff, so the encoding is carried through.
    """
    with open(path, "rb") as handle:
        data = handle.read()

    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16"), "utf-16"

    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig"

    for encoding in ("utf-8", "cp1252"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("utf-8", data, 0, 1, f"no encoding handled {path}")


def rewrite_tokens():
    changed = 0

    for path in walk_text_files():
        text, encoding = read_text(path)

        if OLD not in text:
            continue

        with open(path, "w", encoding=encoding, newline="") as handle:
            handle.write(text.replace(OLD, NEW))

        changed += 1

    print(f"rewrote {changed} files")


def set_package_identity():
    path = os.path.join(NEW_DIR, NEW + ".csproj")

    text, encoding = read_text(path)

    text = text.replace(f"<PackageId>{NEW}</PackageId>", f"<PackageId>{PACKAGE}</PackageId>")
    text = re.sub(r"<RepositoryUrl>[^<]*</RepositoryUrl>", f"<RepositoryUrl>{REPOSITORY}</RepositoryUrl>", text)

    if "Package identifier and assembly name differ" not in text:
        text = text.replace(
            "    <PackageId>",
            "    <!-- Package identifier and assembly name differ on purpose: see tools/rename_to_tactus.py. -->\n"
            "    <PackageId>",
            1,
        )

    with open(path, "w", encoding=encoding, newline="") as handle:
        handle.write(text)

    print(f"package id set to {PACKAGE}, repository to {REPOSITORY}")


def clear_build_output():
    removed = 0

    for base, dirs, _ in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", "venv", "node_modules"}]

        for name in list(dirs):
            if name in {"bin", "obj"}:
                shutil.rmtree(os.path.join(base, name), ignore_errors=True)
                dirs.remove(name)
                removed += 1

    print(f"removed {removed} build directories")


if __name__ == "__main__":
    rename_directory()
    rename_project_file()
    rewrite_tokens()
    set_package_identity()
    clear_build_output()
