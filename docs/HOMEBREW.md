# Homebrew Installation Guide

Complete guide for installing and maintaining Blasphemer via Homebrew.

## For Users

### Install via Homebrew Tap

```bash
# Add the tap
brew tap sunkencity999/blasphemer

# Install blasphemer
brew install blasphemer
```

That's it! Blasphemer and all dependencies are now installed.

### Usage After Install

```bash
# Interactive launcher
blasphemer.sh

# Command line
blasphemer Qwen/Qwen2.5-3B-Instruct

# Convert to GGUF
convert-to-gguf.sh ~/blasphemer-models/my-model
```

### Update

```bash
brew update
brew upgrade blasphemer
```

### Uninstall

```bash
brew uninstall blasphemer
brew untap sunkencity999/blasphemer
```

## For Maintainers

### Creating a Release

1. **Update version in pyproject.toml**
   ```toml
   version = "1.0.2"
   ```

2. **Tag and push**
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 1.0.2"
   git tag -a v1.0.2 -m "Release v1.0.2"
   git push origin master
   git push origin v1.0.2
   ```

3. **Create GitHub Release**
   - Go to https://github.com/sunkencity999/blasphemer/releases/new
   - Select tag: v1.0.2
   - Title: "Blasphemer v1.0.2"
   - Describe changes
   - Publish release

4. **Update Formula SHA256**
   ```bash
   # Download the release tarball
   curl -L https://github.com/sunkencity999/blasphemer/archive/refs/tags/v1.0.2.tar.gz -o blasphemer-1.0.2.tar.gz
   
   # Calculate SHA256
   shasum -a 256 blasphemer-1.0.2.tar.gz
   
   # Update Formula/blasphemer.rb with the hash
   ```

5. **Test the formula**
   ```bash
   brew install --build-from-source ./Formula/blasphemer.rb
   brew test blasphemer
   brew audit --strict blasphemer
   ```

### Setting Up the Tap Repository

The tap should be a separate repository: `homebrew-blasphemer`

**Structure:**
```
homebrew-blasphemer/
├── README.md
└── Formula/
    └── blasphemer.rb
```

**Create the tap repo:**
```bash
# Create new repo on GitHub: homebrew-blasphemer
cd ~/
git clone https://github.com/sunkencity999/homebrew-blasphemer.git
cd homebrew-blasphemer
mkdir Formula
cp ~/blasphemer/Formula/blasphemer.rb Formula/
git add .
git commit -m "Initial Homebrew formula for Blasphemer"
git push origin main
```

### Formula Development Tips

**Test locally:**
```bash
brew install --build-from-source ~/path/to/homebrew-blasphemer/Formula/blasphemer.rb
```

**Debug installation:**
```bash
brew install --verbose --debug blasphemer
```

**Check for issues:**
```bash
brew audit --strict --online blasphemer
```

**Test uninstall:**
```bash
brew uninstall blasphemer
```

### CI/CD for Formula

Add GitHub Actions to `homebrew-blasphemer` repo:

**.github/workflows/tests.yml:**
```yaml
name: brew test-bot
on:
  pull_request:
jobs:
  test-bot:
    runs-on: macos-latest
    steps:
      - name: Set up Homebrew
        uses: Homebrew/actions/setup-homebrew@master
      - name: Run brew test-bot
        run: brew test-bot --only-cleanup-before
```

## Formula Details

### Dependencies

- **cmake** (build time) - For building llama.cpp
- **python@3.12** (runtime) - Python interpreter

### What Gets Installed

- `/opt/homebrew/bin/blasphemer` - Main CLI tool
- `/opt/homebrew/bin/blasphemer.sh` - Interactive launcher
- `/opt/homebrew/bin/convert-to-gguf.sh` - GGUF converter
- `/opt/homebrew/var/blasphemer-models/` - Default models directory
- `/opt/homebrew/Cellar/blasphemer/VERSION/` - Installation directory

### Post-Install

The formula creates:
- Virtual environment with all Python dependencies
- Compiled llama.cpp binaries with Metal support
- Helper scripts in PATH
- Default models directory

## Alternative: Direct Source Install

If you prefer not to use Homebrew:

```bash
# Use the automated installer
curl -fsSL https://raw.githubusercontent.com/sunkencity999/blasphemer/master/install-macos.sh | bash
```

## Support

- Issues: https://github.com/sunkencity999/blasphemer/issues
- Formula issues: https://github.com/sunkencity999/homebrew-blasphemer/issues
