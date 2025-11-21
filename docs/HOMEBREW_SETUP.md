# Homebrew Tap Setup Guide

Step-by-step guide to publish Blasphemer via Homebrew.

## Quick Overview

You'll create two repositories:
1. **blasphemer** (already exists) - The main project
2. **homebrew-blasphemer** (new) - The Homebrew tap

## Step 1: Create the Tap Repository

### On GitHub

1. Go to https://github.com/new
2. **Repository name**: `homebrew-blasphemer` (must start with "homebrew-")
3. **Description**: "Homebrew tap for Blasphemer - LLM decensoring tool for macOS"
4. **Public** repository
5. **Do NOT** initialize with README (we'll create it)
6. Click "Create repository"

### Locally

```bash
cd ~/
mkdir homebrew-blasphemer
cd homebrew-blasphemer

# Initialize git
git init
git branch -M main

# Create directory structure
mkdir Formula

# Copy the formula
cp ~/blasphemer/Formula/blasphemer.rb Formula/

# Create README
cat > README.md << 'EOF'
# Blasphemer Homebrew Tap

Homebrew tap for [Blasphemer](https://github.com/sunkencity999/blasphemer) - Enhanced LLM decensoring tool optimized for macOS (Apple Silicon).

## Installation

```bash
brew tap sunkencity999/blasphemer
brew install blasphemer
```

## Usage

```bash
# Interactive launcher
blasphemer.sh

# Command line
blasphemer Qwen/Qwen2.5-3B-Instruct

# Convert to GGUF
convert-to-gguf.sh ~/blasphemer-models/my-model
```

## Documentation

- Main Project: https://github.com/sunkencity999/blasphemer
- User Guide: https://github.com/sunkencity999/blasphemer/blob/master/USER_GUIDE.md
- Homebrew Guide: https://github.com/sunkencity999/blasphemer/blob/master/HOMEBREW.md

## Updating

```bash
brew update
brew upgrade blasphemer
```

## Issues

Report issues at: https://github.com/sunkencity999/blasphemer/issues
EOF

# Commit and push
git add .
git commit -m "Initial Homebrew tap for Blasphemer"
git remote add origin git@github.com:sunkencity999/homebrew-blasphemer.git
git push -u origin main
```

## Step 2: Create First Release

### In the blasphemer repository

```bash
cd ~/blasphemer

# Make sure everything is committed
git status

# Tag the release
git tag -a v1.0.1 -m "Release v1.0.1 - Stable with testing suite

Features:
- Comprehensive test suite (34 tests passing)
- Interactive launcher script
- GGUF conversion script
- Checkpoint/resume system
- Apple Silicon MPS support
- LM Studio integration

Tested on:
- macOS Sonoma/Sequoia
- Apple Silicon (M1/M2/M3)
- Python 3.10-3.14"

# Push the tag
git push origin v1.0.1
```

### On GitHub

1. Go to https://github.com/sunkencity999/blasphemer/releases/new
2. **Choose a tag**: v1.0.1
3. **Release title**: "Blasphemer v1.0.1 - Stable Release"
4. **Description**:
   ```markdown
   ## Blasphemer v1.0.1 - Production Ready
   
   First stable release of Blasphemer, the macOS-optimized LLM decensoring tool.
   
   ### ✨ Key Features
   
   - **Apple Silicon Support**: Native MPS GPU acceleration
   - **Checkpoint System**: Automatic resume after interruptions
   - **LM Studio Integration**: One-command GGUF conversion
   - **Interactive Launcher**: Menu-driven interface (`blasphemer.sh`)
   - **Testing Suite**: 34 passing tests ensuring reliability
   
   ### 📦 Installation
   
   **Homebrew (Recommended):**
   ```bash
   brew tap sunkencity999/blasphemer
   brew install blasphemer
   ```
   
   **Automated Installer:**
   ```bash
   curl -fsSL https://raw.githubusercontent.com/sunkencity999/blasphemer/master/install-macos.sh | bash
   ```
   
   ### 📚 Documentation
   
   - [User Guide](USER_GUIDE.md)
   - [Homebrew Guide](HOMEBREW.md)
   - [Testing Guide](TESTING.md)
   
   ### 🧪 Tested On
   
   - macOS Sonoma 14.x
   - macOS Sequoia 15.x
   - Apple Silicon (M1, M2, M3, M4)
   - Python 3.10, 3.11, 3.12, 3.14
   
   ### 🙏 Credits
   
   Based on [Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann
   ```

5. Click "Publish release"

## Step 3: Update Formula with SHA256

```bash
# Download the release tarball
cd ~/Downloads
curl -L https://github.com/sunkencity999/blasphemer/archive/refs/tags/v1.0.1.tar.gz -o blasphemer-1.0.1.tar.gz

# Calculate SHA256
shasum -a 256 blasphemer-1.0.1.tar.gz
# Copy the hash that's printed

# Update the formula
cd ~/homebrew-blasphemer
nano Formula/blasphemer.rb
# Replace the empty sha256 "" with the actual hash
# sha256 "abc123def456..."

# Commit and push
git add Formula/blasphemer.rb
git commit -m "Update formula with v1.0.1 SHA256"
git push origin main
```

## Step 4: Test the Formula

```bash
# Test installation from source
brew install --build-from-source ~/homebrew-blasphemer/Formula/blasphemer.rb

# Test the installation
blasphemer --help
blasphemer.sh --help
convert-to-gguf.sh --help

# Run tests
brew test blasphemer

# Audit the formula
brew audit --strict blasphemer

# If everything works, uninstall test version
brew uninstall blasphemer
```

## Step 5: Test from Tap

```bash
# Add your tap
brew tap sunkencity999/blasphemer

# Install from tap
brew install blasphemer

# Test again
blasphemer --help

# Success! Your formula is live!
```

## Step 6: Announce

Update README.md in blasphemer repo to highlight Homebrew installation.

Post on:
- GitHub Discussions
- Reddit (r/MachineLearning, r/LocalLLaMA)
- Twitter/X

Example announcement:
```
🎉 Blasphemer v1.0.1 is now available via Homebrew!

Decensor LLMs on macOS with Apple Silicon support:
brew tap sunkencity999/blasphemer
brew install blasphemer

Features:
✅ Native MPS GPU support
✅ Auto checkpoint/resume
✅ LM Studio GGUF conversion
✅ Interactive menu interface

GitHub: https://github.com/sunkencity999/blasphemer
```

## Maintenance

### When you release a new version:

1. Update version in `pyproject.toml`
2. Commit: `git commit -am "Bump version to 1.0.2"`
3. Tag: `git tag -a v1.0.2 -m "Release v1.0.2"`
4. Push: `git push origin master && git push origin v1.0.2`
5. Create GitHub release
6. Calculate new SHA256
7. Update formula in homebrew-blasphemer
8. Commit and push formula
9. Test: `brew upgrade blasphemer`

### Users will update with:

```bash
brew update
brew upgrade blasphemer
```

## Troubleshooting

**Formula audit fails?**
```bash
brew audit --new-formula blasphemer
# Fix issues shown
```

**Installation fails?**
```bash
brew install --verbose --debug blasphemer
# Review error messages
```

**Need to test locally?**
```bash
brew install --build-from-source ./Formula/blasphemer.rb
```

## Success Criteria

✅ `brew install blasphemer` works  
✅ `blasphemer --help` shows help  
✅ `brew test blasphemer` passes  
✅ `brew audit blasphemer` passes  
✅ Scripts are in PATH  
✅ llama.cpp is built with Metal support  

## Next Steps

Once stable, consider:
- Submit to `homebrew-core` (more visibility, but stricter requirements)
- Add to awesome-macos lists
- Create video tutorial
- Write blog post

Congratulations! 🎉 Blasphemer is now easily installable via Homebrew!
