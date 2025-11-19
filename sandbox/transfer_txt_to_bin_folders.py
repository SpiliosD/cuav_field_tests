"""Transfer .txt files to folders containing matching .bin files.

This script scans a target directory tree for .bin files, indexes them,
and then copies/moves matching .txt files from a source directory to
all folders where their corresponding .bin files exist.

Example:
    # Copy .txt files to folders with matching .bin files
    python transfer_txt_to_bin_folders.py \
        --source "C:/Users/User/Desktop/TXT_Files" \
        --target "G:/Raymetrics_Tests/BOMA2025" \
        --copy

    # Move .txt files instead of copying
    python transfer_txt_to_bin_folders.py \
        --source "C:/Users/User/Desktop/TXT_Files" \
        --target "G:/Raymetrics_Tests/BOMA2025" \
        --move
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable=None, **kwargs):
        if iterable is None:
            class FakeProgressBar:
                def update(self, n=1): pass
                def close(self): pass
            return FakeProgressBar()
        return iterable


def index_bin_files(root_dir: Path) -> DefaultDict[str, list[Path]]:
    """
    Walk through root_dir and build a mapping: base_name → [folders containing name.bin].

    Parameters
    ----------
    root_dir : Path
        Root directory to search for .bin files.

    Returns
    -------
    DefaultDict[str, list[Path]]
        Dictionary mapping lowercase base filename to list of directories
        containing that .bin file.

    Examples
    --------
    >>> bin_locations = index_bin_files(Path("G:/Raymetrics_Tests"))
    >>> # If "spectra_2025-01-01_12-00-00.12.bin" exists in "folder1" and "folder2":
    >>> # bin_locations["spectra_2025-01-01_12-00-00.12"] == [Path("folder1"), Path("folder2")]
    """
    bin_locations: DefaultDict[str, list[Path]] = defaultdict(list)

    if not root_dir.exists():
        raise FileNotFoundError(f"Target root directory does not exist: {root_dir}")

    if not root_dir.is_dir():
        raise NotADirectoryError(f"Target root is not a directory: {root_dir}")

    # Walk through directory tree
    for bin_file in root_dir.rglob("*.bin"):
        if bin_file.is_file():
            # Use lowercase base name as key for case-insensitive matching
            base_name = bin_file.stem.lower()
            bin_locations[base_name].append(bin_file.parent)

    return bin_locations


def transfer_txt_files(
    source_dir: Path,
    bin_locations: DefaultDict[str, list[Path]],
    move: bool = False,
    verbose: bool = True,
) -> tuple[int, int, int]:
    """
    Copy/move each .txt file to every folder where its .bin counterpart exists.

    Parameters
    ----------
    source_dir : Path
        Directory containing .txt files to transfer.
    bin_locations : DefaultDict[str, list[Path]]
        Dictionary mapping base filenames to list of destination directories.
    move : bool
        If True, move files instead of copying. Default: False (copy).
    verbose : bool
        If True, print progress and statistics. Default: True.

    Returns
    -------
    tuple[int, int, int]
        (total_files, transferred_files, skipped_files)
        - total_files: Total .txt files found
        - transferred_files: Files successfully transferred
        - skipped_files: Files with no matching .bin files

    Notes
    -----
    - If move=True and a .txt file matches multiple .bin files in different
      folders, it will be moved to the first folder and copied to the rest.
    - Existing files in destination directories will be overwritten.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {source_dir}")

    # Find all .txt files in source directory
    txt_files = [f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]

    if verbose:
        print(f"Found {len(txt_files)} .txt file(s) to process.\n")

    if not txt_files:
        return 0, 0, 0

    transferred_count = 0
    skipped_count = 0
    error_count = 0

    # Process files with progress bar
    pbar = tqdm(total=len(txt_files), desc="Transferring", unit="file")

    for txt_file in txt_files:
        # Use lowercase base name for case-insensitive matching
        base_name = txt_file.stem.lower()
        key = base_name

        if key not in bin_locations:
            # No matching bin file found — skip
            skipped_count += 1
            pbar.update(1)
            continue

        dest_dirs = bin_locations[key]
        file_transferred = False

        for idx, dest_dir in enumerate(dest_dirs):
            dest_path = dest_dir / txt_file.name

            try:
                # Ensure destination directory exists
                dest_dir.mkdir(parents=True, exist_ok=True)

                # If moving and this is the first destination, move the file
                # For subsequent destinations, copy (file already moved)
                if move and idx == 0:
                    if dest_path.exists():
                        dest_path.unlink()  # Remove existing file
                    shutil.move(str(txt_file), str(dest_path))
                    file_transferred = True
                else:
                    # Copy file (or copy after move)
                    shutil.copy2(str(txt_file), str(dest_path))
                    file_transferred = True

            except Exception as e:
                error_count += 1
                if verbose:
                    print(f"\n⚠ ERROR transferring {txt_file.name} to {dest_dir}: {e}", file=sys.stderr)
                continue

        if file_transferred:
            transferred_count += 1

        pbar.update(1)

    pbar.close()

    if verbose:
        print(f"\nTransfer complete:")
        print(f"  Total files: {len(txt_files)}")
        print(f"  Transferred: {transferred_count}")
        print(f"  Skipped (no match): {skipped_count}")
        if error_count > 0:
            print(f"  Errors: {error_count}")

    return len(txt_files), transferred_count, skipped_count


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Transfer .txt files to folders containing matching .bin files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Copy .txt files to folders with matching .bin files
  python transfer_txt_to_bin_folders.py \\
      --source "C:/Users/User/Desktop/TXT_Files" \\
      --target "G:/Raymetrics_Tests/BOMA2025" \\
      --copy

  # Move .txt files instead of copying
  python transfer_txt_to_bin_folders.py \\
      --source "C:/Users/User/Desktop/TXT_Files" \\
      --target "G:/Raymetrics_Tests/BOMA2025" \\
      --move

  # Quiet mode (no progress bar)
  python transfer_txt_to_bin_folders.py \\
      --source "C:/Users/User/Desktop/TXT_Files" \\
      --target "G:/Raymetrics_Tests/BOMA2025" \\
      --copy \\
      --quiet

Note:
  - Files are matched by base name (case-insensitive)
  - If a .txt file matches multiple .bin files in different folders,
    it will be copied/moved to all matching folders
  - When using --move, the file is moved to the first match and
    copied to subsequent matches
  - Existing files in destination directories will be overwritten
        """,
    )

    parser.add_argument(
        "--source",
        "-s",
        type=Path,
        required=True,
        help="Source directory containing .txt files to transfer",
    )

    parser.add_argument(
        "--target",
        "-t",
        type=Path,
        required=True,
        help="Target root directory to search for .bin files",
    )

    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--copy",
        action="store_true",
        help="Copy .txt files to destination folders (default behavior)",
    )

    action_group.add_argument(
        "--move",
        action="store_true",
        help="Move .txt files to destination folders (file moved to first match, copied to rest)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output and statistics",
    )

    args = parser.parse_args()

    # Determine move/copy mode
    move_mode = args.move

    try:
        # Validate and resolve paths
        source_dir = args.source.expanduser().resolve()
        target_dir = args.target.expanduser().resolve()

        if not args.quiet:
            print("=" * 70)
            print("Transfer .txt Files to Matching .bin Folders")
            print("=" * 70)
            print()
            print(f"Source directory: {source_dir}")
            print(f"Target root: {target_dir}")
            print(f"Mode: {'MOVE' if move_mode else 'COPY'}")
            print()

        # Index .bin files
        if not args.quiet:
            print(f"Indexing .bin files in: {target_dir}...")
        bin_locations = index_bin_files(target_dir)

        if not args.quiet:
            total_bin_files = sum(len(dirs) for dirs in bin_locations.values())
            print(f"Found {total_bin_files} .bin file(s) in {len(bin_locations)} unique base name(s).\n")

        # Transfer .txt files
        if not args.quiet:
            print(f"Processing .txt files in: {source_dir}...")
        total, transferred, skipped = transfer_txt_files(
            source_dir,
            bin_locations,
            move=move_mode,
            verbose=not args.quiet,
        )

        if not args.quiet:
            print("\n✓ Done!")
            return 0
        else:
            return 0

    except FileNotFoundError as e:
        print(f"✗ ERROR: {e}", file=sys.stderr)
        return 1
    except NotADirectoryError as e:
        print(f"✗ ERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ ERROR: Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
