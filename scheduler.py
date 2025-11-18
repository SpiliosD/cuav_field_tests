"""Git auto-commit and push scheduler.

This script automatically commits and pushes changes to the repository
every two hours (configurable).
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Try to import schedule library, fallback to simple loop if not available
try:
    import schedule
    USE_SCHEDULE_LIB = True
except ImportError:
    USE_SCHEDULE_LIB = False
    print("Warning: 'schedule' library not found. Using simple time.sleep() loop.")
    print("Install it with: pip install schedule")
    print("Continuing with simple scheduler...\n")


def run_git_command(command: list[str]) -> tuple[bool, str, str]:
    """
    Run a git command and return the result.

    Parameters
    ----------
    command
        List of command arguments (e.g., ['git', 'status'])

    Returns
    -------
    success
        True if command succeeded (exit code 0)
    stdout
        Standard output from the command
    stderr
        Standard error from the command
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).parent,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def check_for_changes() -> bool:
    """Check if there are uncommitted changes in the repository."""
    success, stdout, _ = run_git_command(["git", "status", "--porcelain"])
    if not success:
        return False
    return bool(stdout.strip())


def commit_and_push() -> bool:
    """
    Commit all changes and push to remote repository.

    Returns
    -------
    bool
        True if commit and push were successful
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n[{timestamp}] Starting auto-commit and push...")

    # Check for changes
    if not check_for_changes():
        print("[{timestamp}] No changes to commit.")
        return True

    print(f"[{timestamp}] Changes detected. Staging files...")

    # Stage all changes
    success, stdout, stderr = run_git_command(["git", "add", "."])
    if not success:
        print(f"[{timestamp}] ERROR: Failed to stage files")
        print(f"stderr: {stderr}")
        return False

    # Commit with timestamp
    commit_message = f"Auto-commit: {timestamp}"
    success, stdout, stderr = run_git_command(
        ["git", "commit", "-m", commit_message],
    )
    if not success:
        if "nothing to commit" in stderr.lower():
            print(f"[{timestamp}] Nothing to commit (already staged?).")
            return True
        print(f"[{timestamp}] ERROR: Failed to commit")
        print(f"stderr: {stderr}")
        return False

    print(f"[{timestamp}] Committed successfully: {commit_message}")

    # Push to remote
    print(f"[{timestamp}] Pushing to remote...")
    success, stdout, stderr = run_git_command(["git", "push"])
    if not success:
        print(f"[{timestamp}] ERROR: Failed to push")
        print(f"stderr: {stderr}")
        return False

    print(f"[{timestamp}] ✓ Successfully pushed to remote repository!")
    return True


def run_with_schedule_lib(interval_hours: int = 2):
    """Run scheduler using the schedule library."""
    schedule.every(interval_hours).hours.do(commit_and_push)

    print(f"Git auto-commit scheduler started!")
    print(f"Will commit and push every {interval_hours} hours.")
    print("Press Ctrl+C to stop.\n")

    # Run once immediately (optional - comment out if not desired)
    commit_and_push()

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\n\nScheduler stopped by user.")


def run_with_simple_loop(interval_hours: int = 2):
    """Run scheduler using a simple time.sleep() loop."""
    interval_seconds = interval_hours * 3600

    print(f"Git auto-commit scheduler started!")
    print(f"Will commit and push every {interval_hours} hours.")
    print("Press Ctrl+C to stop.\n")

    # Run once immediately (optional - comment out if not desired)
    commit_and_push()

    try:
        while True:
            time.sleep(interval_seconds)
            commit_and_push()
    except KeyboardInterrupt:
        print("\n\nScheduler stopped by user.")


def main():
    """Main entry point for the scheduler."""
    # Configuration
    INTERVAL_HOURS = 2

    print("=" * 70)
    print("Git Auto-Commit and Push Scheduler")
    print("=" * 70)

    if USE_SCHEDULE_LIB:
        run_with_schedule_lib(INTERVAL_HOURS)
    else:
        run_with_simple_loop(INTERVAL_HOURS)


if __name__ == "__main__":
    main()

