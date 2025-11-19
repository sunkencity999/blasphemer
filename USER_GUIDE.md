# Blasphemer User Guide

Comprehensive guide for installing, configuring, and using Blasphemer to decensor language models on macOS (Apple Silicon).

Blasphemer is an enhanced fork of Heretic, optimized specifically for macOS with significant improvements to stability, user experience, and functionality.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Model Recommendations](#model-recommendations)
5. [LM Studio Integration](#lm-studio-integration)
6. [Checkpoint & Resume System](#checkpoint--resume-system)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)
10. [Resources](#resources)

---

## Introduction

Blasphemer is an automatic censorship removal tool for transformer-based large language models. It uses activation engineering to identify and remove safety training without degrading model capabilities.

### What Makes Blasphemer Different

Blasphemer is an enhanced fork of Heretic with significant improvements:

- **Interactive Launcher**: User-friendly menu-driven interface (`blasphemer.sh`)
- **Homebrew Installation**: First-class macOS package management support
- **Fixed Bugs**: Resolved JSON serialization, bash compatibility, and progress display issues
- **Better UX**: Clear prompts, progress indicators, and error messages
- **Production Ready**: Extensively tested on Apple Silicon with real-world models

### System Information

- **Python Version**: 3.14.0+
- **PyTorch Version**: 2.9.1+ (with MPS support for Apple Silicon)
- **Blasphemer Version**: 1.0.1.post1
- **GPU**: Apple Silicon (Metal Performance Shaders)
- **Platform**: macOS (Apple Silicon optimized)
- **Shell**: Bash 3.2+ compatible

### What Blasphemer Does

Blasphemer automatically:

- Downloads models from Hugging Face
- Calculates refusal directions using activation engineering
- Optimizes abliteration parameters through Bayesian optimization (200 trials)
- Evaluates model performance (KL divergence and refusal rate)
- Presents Pareto-optimal results for selection
- Converts models to GGUF format for local use
- Saves checkpoints for safe interruption and resumption

---

## Installation

### Recommended: Homebrew Installation

The easiest way to install Blasphemer:

```bash
# Add the Homebrew tap
brew tap sunkencity999/blasphemer

# Install Blasphemer
brew install blasphemer

# Run the installer
blasphemer-install
```

The installer will:

- Set up Python virtual environment
- Install all dependencies
- Build llama.cpp with Metal support
- Configure the system for optimal performance

### Manual Installation

Alternatively, install from source:

```bash
# Clone the repository
git clone https://github.com/sunkencity999/blasphemer.git
cd blasphemer

# Run the macOS installer
bash install-macos.sh
```

### Activating the Environment

Before using Blasphemer, activate the virtual environment:

```bash
cd ~/blasphemer
source venv/bin/activate
```

### Verifying Installation

```bash
# Check Blasphemer is installed
blasphemer --help

# Verify GPU availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Use the interactive launcher
./blasphemer.sh
```

Expected output: `MPS available: True`

### Deactivating the Environment

When finished:

```bash
deactivate
```

---

## Quick Start

### Interactive Launcher (Recommended)

The easiest way to use Blasphemer:

```bash
cd ~/blasphemer
./blasphemer.sh
```

The interactive menu guides you through:
1. Model selection (with recommendations)
2. Save location configuration
3. Advanced options (trials, batch size)
4. Quantization level selection
5. Automatic GGUF conversion

### Command Line Workflow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Process model with Blasphemer
blasphemer meta-llama/Llama-3.1-8B-Instruct

# 3. Save model when prompted
# Choose: "Save the model to a local folder"
# Path: ~/blasphemer-models/Llama-3.1-8B-Instruct-blasphemer

# 4. Convert to GGUF for LM Studio
./convert-to-gguf.sh ~/blasphemer-models/Llama-3.1-8B-Instruct-blasphemer
```

### First-Time Test

Start with a small model to verify your setup:

```bash
# Using interactive launcher (recommended)
./blasphemer.sh
# Select option 1, then model 1 (Phi-3-mini)

# Or command line
blasphemer microsoft/Phi-3-mini-4k-instruct
```

Processing time: 15-20 minutes on Apple Silicon

### Basic Commands

```bash
# Interactive launcher (recommended)
./blasphemer.sh

# View all options
blasphemer --help

# Process a specific model
blasphemer <model-name>

# Process with custom trials
blasphemer --n-trials 100 <model-name>

# Resume interrupted processing
blasphemer --model <model-name> --resume true

# Evaluate an existing decensored model
blasphemer --model <original> --evaluate-model <decensored>
```

---

## Model Recommendations

### Best for First-Time Users

#### Phi-3 Mini (3.8B parameters)

```bash
blasphemer microsoft/Phi-3-mini-4k-instruct
```

- Processing time: 15-20 minutes
- Download size: ~7GB
- Best for learning the workflow

#### Qwen 2.5 7B

```bash
blasphemer Qwen/Qwen2.5-7B-Instruct
```

- Processing time: 30-45 minutes
- Excellent quality for size
- Well-tested with Blasphemer

### Recommended Medium Models

#### Llama 3.1 8B (Most Popular)

```bash
blasphemer meta-llama/Llama-3.1-8B-Instruct
```

- Industry standard
- Processing time: 45-60 minutes
- Great Blasphemer results

#### Mistral 7B v0.3

```bash
blasphemer mistralai/Mistral-7B-Instruct-v0.3
```

- High quality output
- Processing time: 40-50 minutes

### Larger Models (Best Quality)

#### Qwen 14B

```bash
blasphemer Qwen/Qwen2.5-14B-Instruct
```

- Processing time: 60-90 minutes
- Excellent quality

#### Llama 3.1 70B

```bash
blasphemer meta-llama/Llama-3.1-70B-Instruct
```

- Highest quality
- Requires significant resources
- Processing time: Several hours

### Models to Avoid

#### Multimodal Models

These require additional dependencies that are difficult to install on macOS:

- ERNIE-4.5-VL (requires decord package)
- LLaVA models (vision processing)
- Whisper variants (audio processing)

**Recommendation**: Start with text-only models.

#### Very Large Models

Unless you have substantial resources (64GB+ RAM):

- Models > 70B parameters
- MoE models with high parameter counts

#### Unsupported Architectures

Blasphemer does not support:

- SSMs/hybrid models (Mamba, etc.)
- Models with inhomogeneous layers
- Certain novel attention mechanisms

### Proven Compatible Models

- Llama 3.1 (all sizes)
- Mistral (all versions)
- Qwen 2.5 (all sizes)
- Gemma 2 (all sizes)
- Phi-3 (all variants)
- Command R (standard version)

### Hardware Recommendations

For Apple Silicon (MPS):

- **Recommended**: 7B-14B models
- **Maximum**: 30B models (with quantization)
- **Processing time**: 2-3x slower than NVIDIA GPU

---

## LM Studio Integration

After Blasphemer processes a model, you can use it in LM Studio by converting to GGUF format.

### llama.cpp Installation

llama.cpp has been installed and configured:

- **Repository**: `/Users/christopher.bradford/blasphemer/llama.cpp`
- **Helper Script**: `/Users/christopher.bradford/blasphemer/convert-to-gguf.sh`
- **Built with**: Metal support for Apple Silicon GPU acceleration

### Converting Models to GGUF

#### Using the Helper Script (Recommended)

```bash
# Default Q4_K_M quantization (good balance)
./convert-to-gguf.sh ~/blasphemer-models/your-model-blasphemer

# Custom quantization level
./convert-to-gguf.sh ~/blasphemer-models/your-model-blasphemer output-name Q5_K_M

# View help
./convert-to-gguf.sh
```

#### Available Quantization Types

| Type | Size (7B) | Quality | Use Case |
|------|-----------|---------|----------|
| Q4_K_M | ~4.5GB | Good | Recommended balance |
| Q5_K_M | ~5.3GB | Better | Higher quality |
| Q8_0 | ~8GB | High | Maximum quality |
| F16 | ~14GB | Full | No quality loss |

**Perplexity Impact**:
- Q4_K_M: +0.18 ppl
- Q5_K_M: +0.06 ppl
- Q8_0: +0.00 ppl (minimal impact)

#### Manual Conversion (Advanced)

If you prefer to use llama.cpp tools directly:

**Step 1: Convert to F16 GGUF**

```bash
source venv/bin/activate
python llama.cpp/convert_hf_to_gguf.py ~/blasphemer-models/your-model \
  --outfile ~/blasphemer-models/your-model-f16.gguf \
  --outtype f16
```

**Step 2: Quantize**

```bash
./llama.cpp/build/bin/llama-quantize \
  ~/blasphemer-models/your-model-f16.gguf \
  ~/blasphemer-models/your-model-Q4_K_M.gguf \
  Q4_K_M
```

### Loading in LM Studio

After conversion:

1. Open LM Studio
2. The `.gguf` file will automatically appear in the models list
3. Or manually load via "Load Model" → select the GGUF file
4. Start using your decensored model

### Alternative: Upload to Hugging Face

During Blasphemer's save prompt:

1. Choose "Upload the model to Hugging Face"
2. In LM Studio:
   - Search for your model by username/model-name
   - Download directly from LM Studio
   - LM Studio handles conversion automatically

---

## Checkpoint & Resume System

The checkpoint system automatically saves optimization progress, allowing you to safely interrupt and resume long-running processes.

### How It Works

**Automatic Checkpointing**:
- Saves progress after every trial to SQLite database
- Location: `.blasphemer_checkpoints/` directory
- Format: `blasphemer_<model>_<hash>.db`
- Overhead: Negligible (~milliseconds per save)

**What's Saved**:
- All completed trial parameters
- Trial results (KL divergence, refusals)
- Optimization state (TPE sampler)
- Model metadata

### Using Checkpoints

#### Starting a New Run

```bash
# Normal operation - checkpoints saved automatically
blasphemer meta-llama/Llama-3.1-8B-Instruct
```

Checkpoint will be saved to:
`.blasphemer_checkpoints/blasphemer_Llama-3.1-8B-Instruct_<hash>.db`

#### Resuming After Interruption

If your run is interrupted (Ctrl+C, power loss, crash):

```bash
# Resume from checkpoint
blasphemer --resume meta-llama/Llama-3.1-8B-Instruct
```

Output will show:

```
Found existing checkpoint: .blasphemer_checkpoints/blasphemer_Llama-3.1-8B-Instruct_<hash>.db
* Completed trials: 87/200
Resuming optimization - 113 trials remaining
```

#### Continuing Across Sessions

You can intentionally stop and resume later:

```bash
# Session 1: Run 50 trials
blasphemer --n-trials 50 model-name

# Later, Session 2: Run 150 more trials
blasphemer --resume --n-trials 200 model-name
```

The system will run 150 additional trials (200 total - 50 already complete).

### Configuration Options

#### Command Line

```bash
# Enable resume
blasphemer --resume <model>

# Custom checkpoint directory
blasphemer --checkpoint-dir /path/to/checkpoints <model>

# Combine options
blasphemer --resume --checkpoint-dir ./my-checkpoints <model>
```

#### Config File

In `config.toml`:

```toml
# Directory for checkpoints
checkpoint_dir = ".blasphemer_checkpoints"

# Auto-resume (always resume if checkpoint exists)
resume = true
```

### Checkpoint Management

#### Viewing Checkpoints

```bash
# List all checkpoints
ls -lh .blasphemer_checkpoints/
```

#### Checkpoint Sizes

- 50 trials: ~4-6 MB
- 100 trials: ~8-12 MB
- 200 trials: ~15-25 MB

#### Cleaning Up

After successful completion, checkpoints can be deleted:

```bash
# Remove all checkpoints
rm -rf .blasphemer_checkpoints/

# Remove specific checkpoint
rm .blasphemer_checkpoints/blasphemer_model-name_*.db
```

### Best Practices

1. **Always enable for long runs**: For runs > 2 hours, checkpoints are essential
2. **Monitor disk space**: Each checkpoint is 15-25 MB
3. **Keep checkpoints during optimization**: Don't delete until finished
4. **Use --resume for interruptions**: Safe to interrupt with Ctrl+C
5. **Backup critical runs**: Copy checkpoint files for important models

### Safety Features

- Each trial saved immediately - no data loss between trials
- Atomic writes - checkpoint never corrupted mid-write
- Version tracking - detects incompatible checkpoints
- Auto-recovery - falls back to fresh start if checkpoint corrupt

**What's Safe**:
- Ctrl+C during trial: Current trial lost, previous trials saved
- Power failure: At most one trial lost
- System crash: Resume from last completed trial
- Kill signal: Checkpoint persists

**What's Not Saved**:
- In-progress trial: Only completed trials are checkpointed
- Loaded model: Model must be reloaded on resume
- Terminal output: Only trial data, not console logs

---

## Configuration

### Configuration Files

- **Default**: `config.default.toml` - Reference configuration
- **Custom**: `config.toml` - Your custom settings (create from default)

### Command Line vs Config File

Settings can be specified via:

1. **Command line flags** (highest priority)
2. **Environment variables** (prefix with `BLASPHEMER_`)
3. **Config file** (`config.toml`)
4. **Defaults** (`config.default.toml`)

### Common Configuration Options

```toml
# Number of optimization trials
n_trials = 200

# Number of exploration trials
n_startup_trials = 60

# Batch size (0 = auto)
batch_size = 0

# Maximum batch size for auto-detection
max_batch_size = 128

# Maximum response length
max_response_length = 100

# Checkpoint directory
checkpoint_dir = ".blasphemer_checkpoints"

# Resume from checkpoint
resume = false

# System prompt
system_prompt = "You are a helpful assistant."
```

### Refusal Markers

Strings that identify a response as a refusal:

```toml
refusal_markers = [
    "sorry",
    "i can't",
    "i cannot",
    "i won't",
    "i will not",
    "i'm unable",
    "i am unable",
    "i'm an ai",
    "i am an ai",
    "as an ai",
    "ai assistant",
    "violat",
    "prohibit",
    "illegal",
    "harmful",
    "inappropriate",
    "unethical",
]
```

### Dataset Configuration

Customize datasets for calculating refusal directions:

```toml
[good_prompts]
dataset = "mlabonne/harmless_alpaca"
split = "train[:400]"
column = "text"

[bad_prompts]
dataset = "mlabonne/harmful_behaviors"
split = "train[:400]"
column = "text"
```

### Viewing All Options

```bash
blasphemer --help
```

---

## Troubleshooting

### Installation Issues

#### Python Import Errors

If you get import errors:

```bash
source venv/bin/activate
pip install --upgrade pip
pip install -e .
```

#### GPU Not Detected

Verify MPS availability:

```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Should output: `MPS available: True`

Note: On some systems, Blasphemer may show "GPU type: Apple Silicon (MPS)" confirming proper detection. Your Apple Silicon GPU will be used for acceleration via Metal Performance Shaders.

### Processing Issues

#### Out of Memory

Reduce batch size:

```bash
blasphemer --max-batch-size 32 your-model
```

Or try a smaller model first.

#### Import Error: Missing Dependencies

Some models require additional packages. If you see import errors:

```bash
source venv/bin/activate
pip install <missing-package>
```

For multimodal models with problematic dependencies, switch to a text-only model instead.

#### Model Loading Fails

Common causes:

1. **Model not found**: Check model ID on Hugging Face
2. **Unsupported architecture**: Try a different model from recommendations
3. **Insufficient memory**: Use a smaller model

### Conversion Issues

#### llama.cpp Python Errors

Reinstall dependencies:

```bash
source venv/bin/activate
pip install sentencepiece gguf protobuf
```

#### Binary Not Found

Rebuild llama.cpp:

```bash
cd llama.cpp
cmake -B build
cmake --build build --config Release --target llama-quantize -j 8
```

#### Conversion Script Fails

Check model path:

```bash
# Verify model directory exists and contains files
ls ~/blasphemer-models/your-model-blasphemer
```

Should show: `config.json`, `tokenizer.json`, model weight files, etc.

### LM Studio Issues

#### Model Not Appearing

1. Check GGUF file exists:
   ```bash
   ls ~/blasphemer-models/*.gguf
   ```

2. Manually load: LM Studio → "Load Model" → select file

3. Restart LM Studio

#### Poor Model Performance

- Try higher quantization (Q5_K_M or Q8_0)
- Verify correct model was converted
- Check LM Studio temperature/sampling settings

### Checkpoint Issues

#### "Could not load checkpoint"

Corrupted database file:

```bash
# Rename corrupted file
mv .blasphemer_checkpoints/checkpoint.db checkpoint.db.backup

# Start fresh
blasphemer your-model
```

#### "Study already completed"

All trials already done:

```bash
# Increase trials to run more
blasphemer --resume --n-trials 300 your-model

# Or start fresh
blasphemer your-model  # Without --resume
```

#### Resume Not Working

Ensure you're using the exact same model identifier:

```bash
# These are different:
blasphemer meta-llama/Llama-3.1-8B-Instruct
blasphemer /local/path/Llama-3.1-8B-Instruct

# Use the same one for resume
```

---

## Advanced Usage

### Custom Dataset

Use your own dataset for refusal detection:

```toml
[bad_prompts]
dataset = "path/to/your/dataset"
split = "train"
column = "prompt"
```

### Evaluating Models

Compare original vs decensored model:

```bash
blasphemer --model meta-llama/Llama-3.1-8B-Instruct \
        --evaluate-model your-username/Llama-3.1-8B-Instruct-blasphemer
```

### Multiple Quantization Levels

Create different versions for different use cases:

```bash
# Fast version
./convert-to-gguf.sh ~/blasphemer-models/model-blasphemer model-q4 Q4_K_M

# Quality version
./convert-to-gguf.sh ~/blasphemer-models/model-blasphemer model-q5 Q5_K_M

# Maximum quality version
./convert-to-gguf.sh ~/blasphemer-models/model-blasphemer model-q8 Q8_0
```

### Programmatic Access to Checkpoints

Checkpoints use Optuna's SQLite storage and can be analyzed programmatically:

```python
import optuna

study = optuna.load_study(
    study_name="blasphemer_model_hash",
    storage="sqlite:///path/to/checkpoint.db"
)

# Access trials
for trial in study.trials:
    print(f"Trial {trial.number}: {trial.value}")

# Get best trials
print(study.best_trials)
```

### Updating llama.cpp

To update to the latest version:

```bash
cd llama.cpp
git pull
cmake --build build --config Release --target llama-quantize -j 8
```

### Environment Variables

Configure via environment variables:

```bash
export BLASPHEMER_N_TRIALS=100
export BLASPHEMER_BATCH_SIZE=64
export BLASPHEMER_CHECKPOINT_DIR="~/my-checkpoints"
blasphemer your-model
```

---

## Resources

### Blasphemer Resources

- **Blasphemer Homepage**: <https://github.com/sunkencity999/blasphemer>
- **Homebrew Tap**: <https://github.com/sunkencity999/homebrew-blasphemer>
- **Interactive Launcher**: `./blasphemer.sh` (menu-driven interface)
- **Example Models**: Coming soon

### Original Heretic Project

Blasphemer is based on Heretic by Philipp Emanuel Weidmann:

- **Heretic Homepage**: <https://github.com/p-e-w/heretic>
- **Research Paper**: <https://arxiv.org/abs/2406.11717>
- **Example Models**: <https://huggingface.co/collections/p-e-w/the-bestiary>

### Related Tools

- **llama.cpp**: <https://github.com/ggerganov/llama.cpp>
- **LM Studio**: <https://lmstudio.ai>
- **Hugging Face**: <https://huggingface.co>

### Documentation

- **GGUF Specification**: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
- **Quantization Methods**: <https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md>
- **llama.cpp Docs**: <https://github.com/ggerganov/llama.cpp/tree/master/docs>

### File Locations

- **Blasphemer Installation**: `~/blasphemer/`
- **Python venv**: `~/blasphemer/venv/`
- **llama.cpp**: `~/blasphemer/llama.cpp/`
- **Interactive Launcher**: `~/blasphemer/blasphemer.sh`
- **Conversion Script**: `~/blasphemer/convert-to-gguf.sh`
- **Default Config**: `~/blasphemer/config.default.toml`
- **Checkpoints**: `~/blasphemer/.blasphemer_checkpoints/`
- **Default Model Dir**: `~/blasphemer-models/`

### Getting Help

```bash
# Interactive launcher (easiest)
./blasphemer.sh

# Blasphemer command line options
blasphemer --help

# View documentation
cat README.md
cat USER_GUIDE.md

# View default configuration
cat config.default.toml

# Conversion script help
./convert-to-gguf.sh

# Homebrew formula
brew info blasphemer
```

---

## Summary

Blasphemer provides a streamlined workflow for removing safety training from language models on macOS:

1. **Install**: Homebrew or manual installation with full macOS optimization
2. **Launch**: Interactive menu-driven interface for easy operation
3. **Process**: Automatic optimization of abliteration parameters (200 trials)
4. **Convert**: Transform to GGUF format for LM Studio with Metal acceleration
5. **Use**: Run decensored models locally

### Key Features

- **Interactive Launcher**: Menu-driven interface with guided workflows
- **Homebrew Support**: First-class macOS package management
- **Automatic Checkpointing**: Interruption-proof operation with safe resume
- **Apple Silicon Optimized**: Full MPS (Metal) GPU support
- **Production Ready**: Fixed JSON serialization, bash 3.2 compatibility
- **Comprehensive Model Support**: Tested with Llama, Mistral, Qwen, Phi-3, Gemma
- **Professional GGUF Conversion**: llama.cpp with Metal acceleration
- **Flexible Configuration**: Command line, config files, or environment variables

### Enhanced Over Heretic

- Fixed critical bugs (JSON serialization, command substitution)
- Added interactive launcher for better UX
- Improved error messages and progress indicators
- Bash 3.2 compatibility for macOS default shell
- Homebrew formula for easy installation
- Extensively tested on Apple Silicon

Start with small models like Phi-3-mini to learn the workflow, then scale up to larger models as needed.
