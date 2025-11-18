# GitHub Repository Setup Instructions

Your project has been initialized with git and is ready to be pushed to GitHub.

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and log in to your account
2. Click the "+" icon in the top right corner and select "New repository"
3. Fill in the repository details:
   - **Repository name**: `cuav-field-tests` (or your preferred name)
   - **Description**: "Data reader package for processing wind lidar data"
   - **Visibility**: Choose Public or Private
   - **Do NOT** initialize with README, .gitignore, or license (we already have these)
4. Click "Create repository"

## Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands to run. Use these commands:

```bash
# Add the remote repository (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/cuav-field-tests.git

# Verify the remote was added
git remote -v

# Push your code to GitHub
git branch -M main
git push -u origin main
```

## Alternative: Using SSH (if you have SSH keys set up)

```bash
# Add the remote repository using SSH
git remote add origin git@github.com:USERNAME/cuav-field-tests.git

# Push your code to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Verify

After pushing, refresh your GitHub repository page. You should see all your files:
- `README.md`
- `main.py`
- `data_reader/` directory with all subdirectories
- `.gitignore`

## Future Updates

To push future changes to GitHub:

```bash
# Stage your changes
git add .

# Commit your changes
git commit -m "Description of your changes"

# Push to GitHub
git push
```

## Troubleshooting

If you get authentication errors:
- **HTTPS**: You may need to use a Personal Access Token instead of a password
  - Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
  - Generate a new token with `repo` permissions
  - Use this token as your password when pushing
- **SSH**: Make sure your SSH key is added to your GitHub account
  - Go to GitHub Settings → SSH and GPG keys
  - Add your public SSH key if not already added

## Repository Structure

```
cuav-field-tests/
├── README.md              # Project documentation
├── main.py                # Main testing script
├── .gitignore            # Git ignore rules
└── data_reader/          # Main package
    ├── __init__.py
    ├── matching/         # Matches processed and raw data
    ├── parsing/          # Parses spectra and log files
    └── processing/       # Filters processed arrays
```

