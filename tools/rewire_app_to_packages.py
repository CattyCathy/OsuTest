#!/usr/bin/env python3
"""Point the application at the library as a package, and remove the library from this repository.

The library now lives in its own repository. The application consumes it the way any other consumer would: by package
reference. Until the packages are on nuget.org they are packed into `packages-local/`, which is committed, so that a
clone of this repository still builds - the same trade as vendoring a binary, and it goes away the moment the packages
are published.

What changes here:

* `OsuTest.Game.csproj` and `OsuTest.Game.Tests.csproj`: the two project references become package references.
* `OsuTest.Game.Tests.csproj`: the two `None` entries for the analysis fixtures go, because the fixtures went with the
  tests.
* `nuget.config`: adds `packages-local` as a source. It does not clear the inherited ones, so nuget.org still works.
* `*.slnf`: the two library projects are removed from the filters.
* `OsuTest.sln`: the two projects are removed.
* `OsuTest/Tactus`, `OsuTest/Tactus.Bass` and `OsuTest/OsuTest.Game.Tests/Analysis` are deleted.

Run it after the library repository exists and builds; it packs what it finds there.
"""

import os
import shutil
import subprocess
import sys

APP = r"D:\Linux\Proj\OsuTest"
REPO = r"D:\Linux\Proj\Tactus"
LOCAL = os.path.join(APP, "packages-local")

PACKAGES = ["Tactus.NET", "Tactus.Bass"]
PROJECTS = ["Tactus", "Tactus.Bass"]

# The package identifier and the project directory differ for the analysis, which is the point of the naming split.
PACK_DIRECTORIES = {"Tactus.NET": "Tactus", "Tactus.Bass": "Tactus.Bass"}

NUGET_CONFIG = """<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <!--
    The library is consumed as a package rather than as a project, because it lives in its own repository. Until it is
    published, the packages are packed into packages-local and committed, so this repository still restores and builds
    from a clean clone. Nothing is cleared here, so nuget.org and the machine-wide sources keep working.
  -->
  <packageSources>
    <add key="local" value="packages-local" />
  </packageSources>
</configuration>
"""

PACKAGE_REFERENCES = """  <ItemGroup Label="Package References">
    <!--
      The beat analysis and its optional BASS decoder, from their own repository. Project references were how this
      repository consumed them until the library was split out; the packages are what a consumer sees, so they are what
      this repository now uses too.
    -->
    <PackageReference Include="Tactus.NET" Version="1.0.0" />
    <PackageReference Include="Tactus.Bass" Version="1.0.0" />
  </ItemGroup>

</Project>"""

REMOVALS = [
    os.path.join(APP, "OsuTest", "Tactus"),
    os.path.join(APP, "OsuTest", "Tactus.Bass"),
    os.path.join(APP, "OsuTest", "OsuTest.Game.Tests", "Analysis"),
]


def pack():
    os.makedirs(LOCAL, exist_ok=True)

    for package in PACKAGES:
        subprocess.run(
            ["dotnet", "pack", PACK_DIRECTORIES[package], "-c", "Release", "-o", LOCAL],
            cwd=REPO,
            check=True,
            stdout=subprocess.DEVNULL,
        )

    print(f"packed {', '.join(PACKAGES)} into {os.path.relpath(LOCAL, APP)}")


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return handle.read()


def write(path, text, encoding="utf-8"):
    with open(path, "w", encoding=encoding, newline="") as handle:
        handle.write(text)


def rewire_project(path):
    text = read(path)
    original = text

    for project in PROJECTS:
        text = text.replace(f'    <ProjectReference Include="..\\{project}\\{project}.csproj" />\n', "")

    # The fixtures moved with the tests; their None entries would fail the copy instead of doing nothing.
    text = text.replace('    <None Include="Analysis\\DesignantTimingPoints.csv" CopyToOutputDirectory="PreserveNewest" />\n', "")
    text = text.replace('    <None Include="Analysis\\LogMelFixture.bin" CopyToOutputDirectory="PreserveNewest" />\n', "")
    text = text.replace(
        "  <ItemGroup>\n    <!-- The beatmap's own timing points, the reference the tempo agreement test measures against. -->\n  </ItemGroup>\n\n",
        "",
    )

    if text == original:
        sys.exit(f"nothing changed in {path}; has it already been rewired?")

    text = text.replace("</Project>", PACKAGE_REFERENCES, 1) if "</Project>" in text else text
    write(path, text)
    print(f"rewired {os.path.relpath(path, APP)}")


def rewire_filters():
    directory = os.path.join(APP, "OsuTest")

    for name in os.listdir(directory):
        if not name.endswith(".slnf"):
            continue

        path = os.path.join(directory, name)
        text = read(path)
        original = text

        for project in PROJECTS:
            text = text.replace(f'      "{project}\\\\{project}.csproj",\n', "")

        if text != original:
            write(path, text)
            print(f"removed the library projects from {name}")


def remove_projects():
    sln = os.path.join(APP, "OsuTest", "OsuTest.sln")

    for project in PROJECTS:
        subprocess.run(
            ["dotnet", "sln", sln, "remove", os.path.join("OsuTest", project, project + ".csproj")],
            cwd=APP,
            stdout=subprocess.DEVNULL,
        )

    print("removed the library projects from the solution")


def delete_sources():
    for path in REMOVALS:
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"deleted {os.path.relpath(path, APP)}")


if __name__ == "__main__":
    pack()
    write(os.path.join(APP, "nuget.config"), NUGET_CONFIG)
    print("wrote nuget.config")
    rewire_project(os.path.join(APP, "OsuTest", "OsuTest.Game", "OsuTest.Game.csproj"))
    rewire_project(os.path.join(APP, "OsuTest", "OsuTest.Game.Tests", "OsuTest.Game.Tests.csproj"))
    rewire_filters()
    remove_projects()
    delete_sources()
