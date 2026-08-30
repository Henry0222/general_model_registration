from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import shutil


DISTRIBUTIONS = ("PySide6", "PySide6-Essentials", "PySide6-Addons", "shiboken6")


def collect_licenses(output: Path) -> int:
    copied = 0

    for distribution_name in DISTRIBUTIONS:
        distribution = metadata.distribution(distribution_name)
        package_name = distribution.metadata["Name"] or distribution_name
        version = distribution.version
        destination_root = output / f"{package_name}-{version}"
        for relative in distribution.files or ():
            parts = [part.lower() for part in Path(str(relative)).parts]
            if "licenses" not in parts:
                continue
            source = Path(distribution.locate_file(relative))
            if not source.is_file():
                continue
            license_index = parts.index("licenses")
            destination = destination_root.joinpath(*Path(str(relative)).parts[license_index + 1 :])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1

    if copied == 0:
        raise RuntimeError("No PySide6/Shiboken6 license files were found.")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy license files shipped by the installed Qt for Python wheels."
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    copied = collect_licenses(args.output)
    print(f"Copied {copied} Qt for Python license files to {args.output}")


if __name__ == "__main__":
    main()
