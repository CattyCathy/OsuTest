#!/usr/bin/env python3
"""Point the repository URL at ParaTactus rather than ParaTactus.NET.

The package keeps the .NET suffix, because that is the identifier people search for and it is the thing the assembly
name is allowed to differ from. The repository is named after the project instead, like the others here (Para,
ParaOsiris), so the URL in the two nuspecs matches where the code actually lives. A nuspec pointing at a repository
that does not exist is worse than either choice.
"""

import io

FILES = [
    r"D:\Linux\Proj\ParaTactus\ParaTactus\ParaTactus.csproj",
    r"D:\Linux\Proj\ParaTactus\ParaTactus.Bass\ParaTactus.Bass.csproj",
]

OLD = "github.com/CattyCathy/ParaTactus.NET"
NEW = "github.com/CattyCathy/ParaTactus"

for path in FILES:
    text = io.open(path, encoding="utf-8").read()
    updated = text.replace(OLD, NEW)

    if updated != text:
        io.open(path, "w", encoding="utf-8", newline="").write(updated)
        print(f"updated {path.rsplit(chr(92), 1)[-1]}")
    else:
        print(f"unchanged {path.rsplit(chr(92), 1)[-1]}")
