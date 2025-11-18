# Auto-Commit Scheduler Setup Guide

This guide explains how to set up the automatic git commit and push scheduler that runs every 2 hours.

## Option 1: Run Manually (For Testing)

Simply run the scheduler script directly:

```bash
python scheduler.py
```

The scheduler will run continuously, committing and pushing changes every 2 hours. Press `Ctrl+C` to stop.

## Option 2: Windows Task Scheduler (Recommended for Continuous Operation)

### Step 1: Create the Scheduled Task

1. Open **Task Scheduler** (search for it in the Windows Start menu)

2. Click **Create Basic Task** in the right panel

3. **General Tab:**
   - Name: `Git Auto-Commit Scheduler`
   - Description: `Automatically commits and pushes changes every 2 hours`
   - Select: "Run whether user is logged on or not"
   - Check "Run with highest privileges" (optional, but recommended)

4. **Trigger Tab:**
   - Begin the task: "On a schedule"
   - Settings:
     - Daily
     - Start date/time: Choose current date and a convenient start time
     - Recur every: 1 day
     - Advanced settings:
       - Check "Repeat task every: 2 hours"
       - For a duration of: Indefinitely

5. **Action Tab:**
   - Action: "Start a program"
   - Program/script: `python` (or full path, e.g., `C:\ProgramData\anaconda3\envs\coherent_doppler_lidar_env\python.exe`)
   - Add arguments: `scheduler.py`
   - Start in: `D:\CursorProjects\cuav_field_tests`

6. **Conditions Tab:**
   - Uncheck "Start the task only if the computer is on AC power" (if desired)
   - Uncheck "Stop if the computer switches to battery power" (if desired)

7. **Settings Tab:**
   - Check "Allow task to be run on demand"
   - Check "Run task as soon as possible after a scheduled start is missed"
   - If the task fails, restart every: 10 minutes (optional)

8. Click **OK** and enter your Windows password if prompted

### Step 2: Verify the Task

1. In Task Scheduler, find your task in the task list
2. Right-click → **Run** to test it immediately
3. Check the "Last Run Result" - it should show "0x0" (success)

### Step 3: Check Task History

1. In Task Scheduler, click on your task
2. Click the **History** tab at the bottom
3. You should see entries showing when the task ran and its status

## Option 3: Using the Batch File

A batch file `run_scheduler.bat` is provided for convenience. You can:

1. **Double-click it** to run the scheduler manually
2. **Schedule it in Task Scheduler** instead of running `python scheduler.py` directly:
   - In Task Scheduler, use `run_scheduler.bat` as the program
   - Start in: `D:\CursorProjects\cuav_field_tests`

## Configuration

### Changing the Interval

To change from 2 hours to a different interval, edit `scheduler.py`:

```python
# In the main() function, change:
INTERVAL_HOURS = 2  # Change this value (e.g., 1, 4, 6, 12, 24)
```

### Disabling Immediate First Run

By default, the scheduler commits and pushes immediately when started. To disable this:

In `scheduler.py`, comment out these lines in both `run_with_schedule_lib()` and `run_with_simple_loop()`:

```python
# Run once immediately (optional - comment out if not desired)
# commit_and_push()
```

## Troubleshooting

### Task Scheduler Error: "The task did not run because the user was not logged on"

**Solution:** In Task Scheduler:
1. Open the task properties
2. General tab → Select "Run whether user is logged on or not"
3. Save and re-enter your password

### Git Authentication Issues

If you get authentication errors:

1. **For HTTPS:** Set up a Personal Access Token:
   - Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Generate a new token with `repo` permissions
   - Use this token as your password

2. **For SSH:** Make sure your SSH key is added to your GitHub account

### Scheduler Not Running

1. Check Task Scheduler history for error messages
2. Manually run `python scheduler.py` to see any errors
3. Check that Python is in your system PATH or use the full path in Task Scheduler

### No Changes Being Committed

This is normal if there are no uncommitted changes. The scheduler checks for changes before committing and will skip if there's nothing new.

## Monitoring

To monitor the scheduler:

1. **Task Scheduler:** Check the History tab for run status
2. **GitHub:** Check your repository's commit history to see auto-commits
3. **Logs:** The scheduler prints status messages to stdout/stderr, which Task Scheduler captures in its history

## Stopping the Scheduler

1. **If running manually:** Press `Ctrl+C`
2. **If running via Task Scheduler:**
   - Open Task Scheduler
   - Find your task
   - Right-click → **Disable** or **Delete**

## Notes

- The scheduler uses git commands, so make sure git is installed and accessible from your PATH
- The scheduler will only commit if there are changes detected
- Commit messages include timestamps for easy tracking
- The scheduler checks for changes before attempting to commit, so it's safe to run frequently

