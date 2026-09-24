#!/usr/bin/env python3
"""Rename the decoder's namespace from Tactus.Bass to Tactus.Decoding, in both repositories.

The package stays `Tactus.Bass`; only the namespace moves. The reason is a collision that has now cost two rounds of
build errors and would have cost anyone else the same:

A namespace that ends in the same word as a type shadows that type for everything declared inside it. With
`namespace Tactus.Bass` in the picture, any file whose own namespace starts with `Tactus` - the adapter itself, and
every test under `Tactus.Tests` - resolves `Bass.Init` and `Bass.CreateStream` to the *namespace* `Tactus.Bass` rather
than to `ManagedBass.Bass`, and fails to compile. A `using ManagedBass;` directive does not help, because members of an
enclosing namespace take precedence over using directives.

Renaming the namespace removes the trap instead of documenting it. Consumers who happen to write their own `Tactus.*`
namespaces, and anyone adding a second decoder behind the same namespace, are then safe by construction.

Both repositories are updated together so they cannot drift apart. `using Tactus.Bass;` appears only inside code, never
in prose about the package, so the replacement is unambiguous.
"""

import os
import sys

ROOTS = [r"D:\Linux\Proj\OsuTest", r"D:\Linux\Proj\Tactus"]

SKIP_DIRS = {"bin", "obj", ".git", "headless", "venv", "__pycache__", "node_modules"}

REPLACEMENTS = [
    ("using Tactus.Bass;", "using Tactus.Decoding;"),
    ("namespace Tactus.Bass", "namespace Tactus.Decoding"),
    ("using BassApi = ManagedBass.Bass;\n", ""),
    ("BassApi.", "Bass."),
    (
        "    /// The alias at the top of the file is not decoration: this namespace ends in the same word as the type it needs,\n"
        "    /// so inside it <c>Bass.CreateStream</c> resolves to <c>Tactus.Bass.CreateStream</c> and does not compile. The\n"
        "    /// alternative was renaming the namespace away from the package name, which is worse than one alias.\n",
        "    /// The namespace here is <c>Tactus.Decoding</c> while the package is <c>Tactus.Bass</c>, and that is not\n"
        "    /// arbitrary. A namespace ending in the same word as a type shadows it for everything declared inside it: with\n"
        "    /// <c>Tactus.Bass</c> present, <c>Bass.CreateStream</c> in this file - and <c>Bass.Init</c> in anything\n"
        "    /// declared under a <c>Tactus.*</c> namespace, which includes the whole test suite - resolves to the namespace\n"
        "    /// rather than to <c>ManagedBass.Bass</c>, and does not compile. A using directive does not help, because\n"
        "    /// members of an enclosing namespace take precedence over using directives.\n",
    ),
]

EXTENSIONS = {".cs", ".csproj", ".md", ".txt"}


def rewrite(path):
    with open(path, "rb") as handle:
        data = handle.read()

    # Byte order marks matter here: PowerShell 5.1 writes UTF-16 for the `>` redirection, so the captured build logs at
    # the repository root are UTF-16 while everything else is UTF-8, and decoding the wrong way fails on the first byte.
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        encoding = "utf-16"
    elif data.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
    else:
        encoding = "utf-8"

    text = data.decode(encoding)
    original = text

    for old, new in REPLACEMENTS:
        text = text.replace(old, new)

    if text == original:
        return False

    with open(path, "w", encoding=encoding, newline="") as handle:
        handle.write(text)

    return True


def main():
    changed = 0

    for root in ROOTS:
        if not os.path.isdir(root):
            sys.exit(f"{root} does not exist")

        for base, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for name in files:
                if os.path.splitext(name)[1].lower() in EXTENSIONS and rewrite(os.path.join(base, name)):
                    print(f"  {os.path.relpath(os.path.join(base, name), root)}")
                    changed += 1

    print(f"rewrote {changed} files across {len(ROOTS)} repositories")


if __name__ == "__main__":
    main()
