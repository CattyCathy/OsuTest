#!/usr/bin/env python3
"""Split the analysis library out of the application repository into a repository of its own.

What moves:

* `OsuTest/Tactus`          -> `Tactus/`          the analysis, package `Tactus.NET`
* `OsuTest/Tactus.Bass`     -> `Tactus.Bass/`     the optional decoder, package `Tactus.Bass`
* `OsuTest/OsuTest.Game.Tests/Analysis` -> `Tactus.Tests/`   every test that is about the library

The tests come along because they are the library's specification rather than the application's: they call the frontend,
the postprocessing and the grid directly and never touch the player. Nothing in that directory references an
application type, which is what makes the move possible in one piece.

History is not carried over. The application repository keeps the whole development history of these files, and this
one starts at the commit where the split happened; the README says so. Rewriting history with a filter to preserve it
across three paths, one of which then has to be moved into place, is more risk than a fresh start is worth for a first
release.

Two things are deliberately *not* copied:

* `AnalysisLibraryBoundaryTest.cs` - half of it checks that the library does not depend on the application, which is
  vacuous in a repository that does not contain the application. A version without that half is written instead.
* `tools/` - the scripts there are records of how this repository was built, including this one, and belong to the
  application repository.

Run from anywhere; paths are absolute. Idempotent in the sense that it refuses to overwrite an existing target.
"""

import os
import shutil
import subprocess
import sys

APP = r"D:\Linux\Proj\OsuTest"
REPO = r"D:\Linux\Proj\Tactus"

CORE = "Tactus"
BASS = "Tactus.Bass"
TESTS = "Tactus.Tests"

SKIP_DIRS = {"bin", "obj", ".git", "__pycache__"}
SKIP_FILES = {"AnalysisLibraryBoundaryTest.cs"}

DIRECTORY_BUILD_PROPS = """<!-- Properties every project here needs. Mirrors the application repository's, without the parts that
     are about being the application. -->
<Project>
  <PropertyGroup>
    <GenerateDocumentationFile>true</GenerateDocumentationFile>
    <NoWarn>$(NoWarn);CS1591</NoWarn>
  </PropertyGroup>
</Project>
"""

GITIGNORE = """# Build output
bin/
obj/

# NuGet packages produced by `dotnet pack`, kept out of the tree; releases are published instead.
artifacts/

# IDE
.vs/
.idea/
*.user
*.suo
"""

README = """# Tactus.NET

Beat tracking for audio, in C#: the beat instants of a track, including where the tempo changes, as a grid a UI can
animate against.

Two packages:

| Package | Project | What it is |
| --- | --- | --- |
| `Tactus.NET` | `Tactus/` | the analysis. Three MIT dependencies and nothing else. |
| `Tactus.Bass` | `Tactus.Bass/` | optional. Decodes a path for you, with BASS, which is **proprietary** — free for non-commercial use only. |

`Tactus.Tests/` holds the suite. It is the specification rather than an afterthought: it pins the log-mel frontend
against torchaudio's own output, holds the model against the published implementation, and measures every stage of the
postprocessing against a beatmap's declared tempo.

Start with [`Tactus/README.md`](Tactus/README.md) — the requirements (a model file is needed, and is not distributed
here), the usage, and the limits of the model that were measured rather than assumed.

## Licence

MIT, copyright (c) 2026 CattyCathy — see [`LICENSE`](LICENSE). `Tactus.Bass` is MIT as source but depends on the
proprietary BASS library; its [`THIRD-PARTY-NOTICES.md`](Tactus.Bass/THIRD-PARTY-NOTICES.md) has those terms, and
`Tactus/THIRD-PARTY-NOTICES.md` covers the rest.

## History

This repository starts at the split. The code, and the whole history of how it got here — the frontend bugs that
nothing local could detect, the postprocessing that was measured and reverted, the limits that were found one at a time
— is in the application repository it grew out of.
"""

BOUNDARY_TEST = '''using System;
using System.Linq;
using NUnit.Framework;
using Tactus;
using Tactus.Bass;

namespace Tactus.Tests
{
    /// <summary>
    /// What the analysis is allowed to depend on, checked against the compiled assembly's own reference list, which is
    /// the one description of its dependencies that cannot drift away from the truth.
    /// </summary>
    /// <remarks>
    /// The analysis used to decode audio itself, through BASS, which is proprietary - free for non-commercial use and
    /// licensed per product otherwise - and it arrived with osu.Framework, FFmpeg, ImageSharp under a non-open-source
    /// split licence, a tablet driver stack, and tens of megabytes of native binaries for platforms this library has
    /// nothing to say about. Someone can put all of that back with one using statement, and it would compile and pass
    /// every other test here, which is exactly why this one exists.
    /// </remarks>
    [TestFixture]
    public class AnalysisLibraryBoundaryTest
    {
        /// <summary>
        /// Assemblies whose terms would spread to whoever uses the analysis, which must therefore not be referenced.
        /// </summary>
        private static readonly string[] proprietary = { "ManagedBass", "osu.Framework", "Bass" };

        [Test]
        public void TestTheAnalysisCarriesNothingProprietary()
        {
            var referenced = references(typeof(BeatGrid).Assembly);

            TestContext.Out.WriteLine("the analysis project references:");
            foreach (string name in referenced)
                TestContext.Out.WriteLine($"  {name}");

            var offenders = referenced
                            .Where(r => proprietary.Any(p => r.Contains(p, StringComparison.OrdinalIgnoreCase)))
                            .ToArray();

            Assert.That(offenders, Is.Empty,
                "the analysis can reach a library whose terms are not MIT, so it cannot be published as one: "
                + string.Join(", ", offenders));
        }

        [Test]
        public void TestDecodingIsOptionalAndSeparate()
        {
            var library = typeof(BeatGrid).Assembly;

            // The seam is part of the analysis; the implementation is not, and the point is that the analysis compiles
            // and works without ever loading the assembly that holds it.
            Assert.That(typeof(IAudioDecoder).Assembly, Is.EqualTo(library), "the decode seam belongs with the analysis");
            Assert.That(typeof(MonoMixdown).Assembly, Is.EqualTo(library), "mixdown is arithmetic and belongs with the analysis");
            Assert.That(typeof(BassAudioDecoder).Assembly, Is.Not.EqualTo(library), "the BASS decoder must not be inside the analysis");

            Assert.That(references(typeof(BassAudioDecoder).Assembly), Does.Contain("Tactus"),
                "the adapter is a consumer of the analysis like any other, not the other way round");
        }

        [Test]
        public void TestAllOfTheAnalysisLivesTogether()
        {
            var library = typeof(BeatGrid).Assembly;

            Assert.That(typeof(BeatGridProvider).Assembly, Is.EqualTo(library), "the provider belongs with the analysis");
            Assert.That(typeof(BeatThisBeatTracker).Assembly, Is.EqualTo(library));
            Assert.That(typeof(BeatTrainRegulariser).Assembly, Is.EqualTo(library));
            Assert.That(typeof(OnsetEnvelope).Assembly, Is.EqualTo(library));
            Assert.That(typeof(BeatGridCache).Assembly, Is.EqualTo(library));
        }

        [Test]
        public void TestBeatsCanBeTakenFromAFileOrFromSamples()
        {
            // Both forms are offered deliberately. A caller with a file should not have to decode it to ask for its
            // beats, and a caller feeding audio as it plays has no file to give. The file form asks for a decoder
            // rather than choosing one, which is what keeps the analysis free of BASS.
            var overloads = typeof(BeatThisBeatTracker)
                            .GetMethods()
                            .Where(m => m.IsPublic && m.Name == nameof(BeatThisBeatTracker.BeatTimes))
                            .ToArray();

            var fromPath = overloads.SingleOrDefault(m => m.GetParameters()[0].ParameterType == typeof(string));
            var fromSamples = overloads.SingleOrDefault(m => m.GetParameters()[0].ParameterType == typeof(float[]));

            Assert.That(fromPath, Is.Not.Null, "a path should be accepted");
            Assert.That(fromSamples, Is.Not.Null, "samples should be accepted");
            Assert.That(fromPath!.GetParameters().Select(p => p.ParameterType), Does.Contain(typeof(IAudioDecoder)),
                "the path form should take a decoder rather than choosing one");
        }

        private static string[] references(System.Reflection.Assembly assembly)
        {
            return assembly.GetReferencedAssemblies()
                           .Select(a => a.Name ?? string.Empty)
                           .OrderBy(n => n)
                           .ToArray();
        }
    }
}
'''

TEST_PROJECT = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup Label="Project">
    <TargetFramework>net10.0</TargetFramework>
    <IsPackable>false</IsPackable>
  </PropertyGroup>

  <!--
    NUnit is referenced explicitly rather than relied on transitively. In the application repository it arrived through
    osu.Framework's own dependency on it, which is not a reason for it to be here.
  -->
  <ItemGroup Label="Package References">
    <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.30.0" />
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="18.8.1" />
    <PackageReference Include="NUnit" Version="4.6.1" />
    <PackageReference Include="NUnit3TestAdapter" Version="6.2.0" />
  </ItemGroup>

  <ItemGroup Label="Project References">
    <ProjectReference Include="..\\Tactus\\Tactus.csproj" />
    <ProjectReference Include="..\\Tactus.Bass\\Tactus.Bass.csproj" />
  </ItemGroup>

  <ItemGroup>
    <!-- The beatmap's own timing points, the reference the tempo agreement test measures against. -->
    <None Include="DesignantTimingPoints.csv" CopyToOutputDirectory="PreserveNewest" />
    <!-- torchaudio's log-mel output for a fixed signal, the reference the frontend test measures against. -->
    <None Include="LogMelFixture.bin" CopyToOutputDirectory="PreserveNewest" />
  </ItemGroup>
</Project>
"""


def copy_tree(source, destination):
    copied = 0

    for base, dirs, files in os.walk(source):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        relative = os.path.relpath(base, source)
        target = destination if relative == "." else os.path.join(destination, relative)
        os.makedirs(target, exist_ok=True)

        for name in files:
            if name in SKIP_FILES:
                continue

            shutil.copy2(os.path.join(base, name), os.path.join(target, name))
            copied += 1

    return copied


def read_text(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return handle.read()


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def rewrite(path, old, new):
    text = read_text(path)

    if old not in text:
        sys.exit(f"{path} does not contain {old!r}")

    write_text(path, text.replace(old, new))


def main():
    if os.path.exists(REPO):
        sys.exit(f"{REPO} already exists; remove it first if you mean to redo the split")

    os.makedirs(REPO)

    core = copy_tree(os.path.join(APP, "OsuTest", CORE), os.path.join(REPO, CORE))
    bass = copy_tree(os.path.join(APP, "OsuTest", BASS), os.path.join(REPO, BASS))
    tests = copy_tree(os.path.join(APP, "OsuTest", "OsuTest.Game.Tests", "Analysis"), os.path.join(REPO, TESTS))

    print(f"copied {core} analysis files, {bass} decoder files, {tests} test files")

    # The tests are no longer the application's tests, and the internals they reach into are granted to the new name.
    for base, dirs, files in os.walk(os.path.join(REPO, TESTS)):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for name in files:
            if name.endswith(".cs"):
                rewrite(os.path.join(base, name), "namespace OsuTest.Game.Tests.Analysis", "namespace Tactus.Tests")

    for project in (CORE, BASS):
        rewrite(
            os.path.join(REPO, project, project + ".csproj"),
            '<InternalsVisibleTo Include="OsuTest.Game.Tests" />',
            '<InternalsVisibleTo Include="Tactus.Tests" />',
        )

    write_text(os.path.join(REPO, "Directory.Build.props"), DIRECTORY_BUILD_PROPS)
    write_text(os.path.join(REPO, ".gitignore"), GITIGNORE)
    write_text(os.path.join(REPO, "README.md"), README)
    write_text(os.path.join(REPO, TESTS, "AnalysisLibraryBoundaryTest.cs"), BOUNDARY_TEST)
    write_text(os.path.join(REPO, TESTS, TESTS + ".csproj"), TEST_PROJECT)
    shutil.copy2(os.path.join(REPO, CORE, "LICENSE"), os.path.join(REPO, "LICENSE"))

    print("wrote the repository's own build files, licence and readme")

    subprocess.run(["git", "init", "-b", "main"], cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "add", "-A"], cwd=REPO, check=True)
    print(f"initialised a repository at {REPO} with everything staged")


if __name__ == "__main__":
    main()
