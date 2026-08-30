from __future__ import annotations

from pathlib import Path
import shutil
import zipfile

from collect_pyside_licenses import collect_licenses
from write_sha256 import sha256_file


VERSION = "1.4.1"


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    dist = project / "dist"
    executable = dist / f"GeneralModelRegistration-v{VERSION}.exe"
    executable.resolve(strict=True)

    release = dist / "release" / f"GeneralModelRegistration-v{VERSION}-win64"
    if release.is_dir():
        shutil.rmtree(release)
    release.mkdir(parents=True, exist_ok=True)

    shutil.copy2(executable, release / executable.name)
    shutil.copy2(project / "PORTABLE_README.txt", release / "使用说明.txt")
    shutil.copy2(project / "LICENSE", release / "LICENSE")
    shutil.copy2(
        project / "THIRD_PARTY_NOTICES.md",
        release / "THIRD_PARTY_NOTICES.md",
    )
    copied = collect_licenses(release / "PySide6-LICENSES")

    archive = dist / f"GeneralModelRegistration-v{VERSION}-win64.zip"
    if archive.is_file():
        archive.unlink()
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as output:
        for source in sorted(path for path in release.rglob("*") if path.is_file()):
            output.write(source, source.relative_to(release).as_posix())

    sidecar = archive.with_name(f"{archive.name}.sha256.txt")
    sidecar.write_text(
        f"{sha256_file(archive)}  {archive.name}\n",
        encoding="ascii",
    )
    print(f"Packaged {archive.name} with {copied} Qt license files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
