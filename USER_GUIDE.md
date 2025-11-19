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

Based on extensive testing, here are models ranked by abliteration success rate:

### ⭐ Highly Recommended (High Success Rate)

#### Llama 3.1 8B Instruct (BEST CHOICE)

```bash
blasphemer meta-llama/Llama-3.1-8B-Instruct
```

- **Success rate**: Excellent (80-90%)
- **Expected refusals**: 2-10% (Very Good to Excellent)
- **Expected KL divergence**: 0.15-0.30
- Processing time: 45-60 minutes
- Most tested, industry standard
- Consistently good results

#### Mistral 7B v0.3

```bash
blasphemer mistralai/Mistral-7B-Instruct-v0.3
```

- **Success rate**: Very Good (70-80%)
- **Expected refusals**: 3-12%
- **Expected KL divergence**: 0.15-0.35
- Processing time: 40-50 minutes
- High quality output
- Well-documented

#### Qwen 2.5 7B Instruct

```bash
blasphemer Qwen/Qwen2.5-7B-Instruct
```

- **Success rate**: Very Good (70-80%)
- **Expected refusals**: 3-15%
- **Expected KL divergence**: 0.20-0.40
- Processing time: 30-45 minutes
- Excellent quality for size
- Fast training

### Larger Models (Best Quality)

#### Qwen 2.5 14B Instruct

```bash
blasphemer Qwen/Qwen2.5-14B-Instruct
```

- **Success rate**: Good (60-70%)
- **Expected refusals**: 5-20%
- Processing time: 60-90 minutes
- Excellent quality

#### Llama 3.1 70B

```bash
blasphemer meta-llama/Llama-3.1-70B-Instruct
```

- Highest quality
- Requires significant resources
- Processing time: Several hours

### ⚠️ Challenging Models (Lower Success Rate)

These models can be abliterated but often produce poor results:

#### Phi-3 Mini (3.8B)

```bash
blasphemer microsoft/Phi-3-mini-4k-instruct
```

- **Success rate**: Poor (20-30%)
- **Expected refusals**: 60-90% (often fails completely)
- **Issue**: Multi-directional safety alignment
- **Why difficult**: Microsoft uses aggressive RLHF across multiple directions
- **Recommendation**: Use Llama 3.1 8B instead for learning

#### Gemma 2 (2B, 9B, 27B)

```bash
blasphemer google/gemma-2-9b-it
```

- **Success rate**: Fair (40-50%)
- **Expected refusals**: 20-40%
- **Issue**: Strong, distributed safety mechanisms
- **Recommendation**: Requires more trials (300-500) and patience

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

### Architecture Compatibility

**Well-Tested (High Success Rate):**
- Llama 3/3.1 (all sizes) ⭐
- Mistral 7B (all versions) ⭐
- Qwen 2.5 (7B, 14B, 32B)

**Tested (Moderate Success):**
- Gemma 2 (requires patience)
- Command R (standard version)
- Yi models (some variants)

**Experimental (Low Success):**
- Phi-3 (all variants)
- Very small models (< 3B)
- Highly aligned corporate models

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

### Uploading to Hugging Face

#### Method 1: During Optimization (SafeTensors)

During Blasphemer's save prompt:

1. Choose "Upload the model to Hugging Face"
2. Enter your HF token when prompted
3. Model uploads in SafeTensors format
4. Others can download and use with transformers library

#### Method 2: Upload GGUF Files (Recommended for LM Studio Users)

Blasphemer includes an `upload_gguf.py` script for uploading quantized GGUF models to Hugging Face. This is ideal for sharing models that work directly in LM Studio.

**Prerequisites:**

```bash
cd ~/blasphemer
source venv/bin/activate
huggingface-cli login  # Enter your token from https://huggingface.co/settings/tokens
```

**Basic Upload:**

```bash
# Upload a single quantized model with auto-generated model card
python upload_gguf.py \
    Llama-3.1-8B-Blasphemer-Q4_K_M.gguf \
    --repo-name "Llama-3.1-8B-Blasphemer-GGUF" \
    --create-card
```

**Upload Multiple Quantizations:**

```bash
# Upload Q4_K_M (most popular - 4.5GB)
python upload_gguf.py \
    Llama-3.1-8B-Blasphemer-Q4_K_M.gguf \
    --repo-name "Llama-3.1-8B-Blasphemer-GGUF" \
    --create-card

# Upload Q5_K_M (higher quality - 5.5GB)
python upload_gguf.py \
    Llama-3.1-8B-Blasphemer-Q5_K_M.gguf \
    --repo-name "Llama-3.1-8B-Blasphemer-GGUF" \
    --message "Add Q5_K_M quantization"

# Upload F16 (full precision - 15GB, for power users)
python upload_gguf.py \
    Llama-3.1-8B-Blasphemer-F16.gguf \
    --repo-name "Llama-3.1-8B-Blasphemer-GGUF" \
    --message "Add F16 full precision version"
```

**What the Script Does:**

- ✅ Creates Hugging Face repository automatically
- ✅ Uploads GGUF files with progress display
- ✅ Generates professional model card with your metrics
- ✅ Includes usage instructions for LM Studio, llama.cpp, and Python
- ✅ Proper credits and citations

**Custom Options:**

```bash
# Specify custom username
python upload_gguf.py \
    model.gguf \
    --repo-name "My-Model-GGUF" \
    --username "my-hf-username"

# Custom commit message
python upload_gguf.py \
    model.gguf \
    --repo-name "My-Model-GGUF" \
    --message "Update with improved quantization"
```

**Model Card Contents:**

The auto-generated model card includes:
- Your actual quality metrics (KL divergence, refusal rate, trial number)
- Quantization comparison table
- Usage examples for LM Studio, llama.cpp, and Python
- Ethical considerations and responsible use guidelines
- Proper citations for Llama, Blasphemer, and research papers

**Accessing Your Model:**

After upload, users can:

1. **In LM Studio:**
   - Search: `YOUR_USERNAME/Model-Name-GGUF`
   - Click download
   - Start using immediately

2. **Direct Download:**
   - Visit: `https://huggingface.co/YOUR_USERNAME/Model-Name-GGUF`
   - Download any quantization
   - Import to LM Studio or llama.cpp

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

---

## Enhanced Observability

Blasphemer provides real-time insight into optimization quality, helping you understand what's happening during long runs and whether you're getting good results.

### What You See During Optimization

#### Real-Time Progress Display

```
╔════════════════════════════════════════════════════════════════╗
║ Blasphemer Optimization Progress                              ║
║ Model: meta-llama/Llama-3.1-8B-Instruct                      ║
╚════════════════════════════════════════════════════════════════╝

Trial 47/200 (23.5%, ~14h 23m remaining)
████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

Current Trial:
  Parameters: attn.o_proj (and others)
  KL Divergence: 0.234
  Refusals: 5

Best Trial So Far:
  Trial: #42
  KL Divergence: 0.198 (▼ improving)
  Refusals: 3/200 (1.5%)
  Quality: ██████████░░░░

Trend: ▼ IMPROVING

Expected Outcome: Very Good - Good balance of quality and safety removal
```

#### What Each Metric Means

**KL Divergence** (Lower is better):
- Measures how different the model's behavior is from the original
- `< 0.15`: Excellent - minimal quality impact
- `0.15-0.25`: Very good - good balance
- `0.25-0.40`: Good - acceptable trade-off
- `0.40-0.60`: Acceptable - noticeable impact
- `> 0.60`: Poor - significant degradation

**Refusals** (Lower is better):
- Number of test prompts the model still refuses
- Out of 200 test prompts by default
- `< 2 (1%)`: Excellent removal of safety alignment
- `2-5 (1-2.5%)`: Very good
- `5-10 (2.5-5%)`: Good
- `> 10 (>5%)`: May need more trials or different approach

**Trend Analysis**:
- `▼ IMPROVING`: Quality getting better over recent trials
- `▬ STABLE`: Quality plateaued (may be done)
- `▲ DEGRADING`: Quality getting worse (unusual)

**Quality Bar**:
- Visual representation of KL divergence
- `██████████`: Excellent quality (low KL)
- `█████░░░░░`: Medium quality
- `░░░░░░░░░░`: Poor quality (high KL)

### Completion Summary

After optimization finishes, you see:

```
╔════════════════════════════════════════════════════════════════╗
║ ✓ Optimization Complete!                                      ║
║ Model: meta-llama/Llama-3.1-8B-Instruct                      ║
╚════════════════════════════════════════════════════════════════╝

Summary:
  Total trials: 200
  Total time: 16h 45m
  Avg per trial: 5m 2s

Best Result:
  Trial: #178
  KL Divergence: 0.143
  Refusals: 2/200 (1.0%)
  Quality: Excellent - High quality with minimal refusals

Top 5 Trials:
┌───────┬─────────┬──────────┬──────────────┐
│ Trial │ KL Div  │ Refusals │ Quality      │
├───────┼─────────┼──────────┼──────────────┤
│ #178⭐│ 0.143   │ 2 (1.0%) │ ██████████░░ │
│ #195  │ 0.156   │ 3 (1.5%) │ █████████░░░ │
│ #142  │ 0.167   │ 2 (1.0%) │ ████████░░░░ │
│ #189  │ 0.172   │ 4 (2.0%) │ ████████░░░░ │
│ #203  │ 0.183   │ 3 (1.5%) │ ███████░░░░░ │
└───────┴─────────┴──────────┴──────────────┘
```

### Understanding Quality Predictions

The system predicts outcome quality based on current results:

- **Excellent**: KL < 0.15, refusals < 1% → High-quality abliteration
- **Very Good**: KL < 0.25, refusals < 2.5% → Production-ready
- **Good**: KL < 0.40, refusals < 5% → Acceptable for most uses
- **Acceptable**: KL < 0.60, refusals < 10% → Noticeable quality impact
- **Poor**: KL > 0.60 or refusals > 10% → Consider different parameters

### When to Stop Early

The observability system helps you decide whether to continue:

**Good signs** (keep going):
- Trend showing `▼ IMPROVING`
- Best trial improving every 5-10 trials
- KL divergence decreasing
- Refusals decreasing

**Signs to consider stopping**:
- Trend showing `▬ STABLE` for 30+ trials
- No improvement in best trial for 50+ trials
- Already achieved "Excellent" quality
- Time constraints (you have good-enough results)

**Warning signs**:
- Trend showing `▲ DEGRADING` consistently
- KL divergence increasing over time
- Refusals increasing
- Quality predictions getting worse

### Benefits of Enhanced Observability

1. **Confidence During Long Runs**: Know if it's working without waiting 16 hours
2. **Early Quality Assessment**: Predict final outcome quality early
3. **Informed Decisions**: Stop early if you've achieved your goal
4. **Troubleshooting**: Identify if something is wrong (degrading trend)
5. **Learning**: Understand what good results look like for different models

### Technical Details

The progress tracker:
- Analyzes last 10 trials for trend detection
- Weights KL divergence (60%) and refusals (40%) for overall quality score
- Compares first half vs second half of recent trials for trend direction
- Provides quality predictions based on research-backed thresholds
- Updates after every trial completion

All metrics are also saved to the checkpoint database for later analysis.

---

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

## Testing & Development

Blasphemer includes a comprehensive test suite to ensure reliability, prevent regressions, and validate all critical bug fixes.

### Test Suite Organization

```
tests/
├── __init__.py                          # Test package initialization
├── unit/                                # Unit tests for individual components
│   ├── test_serialization.py          # JSON serialization fixes
│   ├── test_utils.py                   # Utility functions
│   └── test_config.py                  # Configuration handling
├── integration/                         # Integration tests
│   └── test_checkpoint_system.py       # Checkpoint & resume functionality
└── scripts/                             # Shell script validation
    └── test_shell_scripts.sh           # Bash script tests
```

### Running Tests

#### Prerequisites

```bash
# Activate virtual environment
cd ~/blasphemer
source venv/bin/activate

# Install pytest (if not already installed)
pip install pytest

# Optional: Install coverage tools
pip install pytest-cov
```

#### Running All Tests

```bash
# Run complete test suite with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/heretic --cov-report=html

# Run with detailed output
pytest tests/ -vv

# Run tests and stop on first failure
pytest tests/ -x
```

#### Running Specific Test Suites

```bash
# Unit tests only
pytest tests/unit/ -v

# JSON serialization tests (critical bug fixes)
pytest tests/unit/test_serialization.py -v

# Utility function tests
pytest tests/unit/test_utils.py -v

# Checkpoint system integration tests
pytest tests/integration/test_checkpoint_system.py -v

# Run specific test class
pytest tests/unit/test_serialization.py::TestAbliterationParametersSerialization -v

# Run specific test function
pytest tests/unit/test_serialization.py::TestAbliterationParametersSerialization::test_parameters_dict_json_serializable -v
```

#### Shell Script Tests

```bash
# Run bash script validation
bash tests/scripts/test_shell_scripts.sh

# Tests syntax, executability, branding, and error handling
```

### Test Coverage

#### Unit Tests (`tests/unit/`)

**test_serialization.py** - Critical bug fixes (10 tests):
- `TestAbliterationParametersSerialization` (6 tests)
  - Validates dataclass structure
  - Tests `asdict()` conversion to dict
  - Confirms dicts are JSON serializable
  - Proves raw dataclass raises TypeError (the bug)
  - Tests multi-component serialization
  - Validates Optuna trial storage format
- `TestResumeArgument` (2 tests)
  - Validates `--resume true` flag format
  - Tests shell script command construction
- `TestConfigSettings` (2 tests)
  - Validates checkpoint directory naming
  - Tests BLASPHEMER_ environment prefix

**test_utils.py** - Utility functions (15 tests):
- `TestFormatDuration` (5 tests)
  - Seconds, minutes, hours formatting
  - Zero duration handling
  - Float duration rounding
- `TestBatchify` (6 tests)
  - Exact and uneven batches
  - Single batch, empty list
  - Batch size of 1
- `TestEmptyCache` (3 tests)
  - CUDA cache clearing (skipped on Apple Silicon)
  - MPS cache clearing (Apple Silicon)
  - No-GPU fallback
- `TestTrialFormatting` (2 tests)
  - README intro generation with proper attribution
  - Markdown link formatting

**test_config.py** - Configuration management:
- Environment variable handling
- Config file parsing
- Default value validation

#### Integration Tests (`tests/integration/`)

**test_checkpoint_system.py** - Checkpoint functionality (10 tests):
- `TestCheckpointSystem` (4 tests)
  - Checkpoint directory creation
  - Naming convention (blasphemer_ prefix)
  - SQLite database creation and persistence
  - Resume detection logic
- `TestEnvironmentVariables` (2 tests)
  - BLASPHEMER_ prefix usage
  - Checkpoint directory override via env
- `TestModelNaming` (2 tests)
  - -blasphemer suffix validation
  - GGUF naming convention

#### Shell Script Tests (`tests/scripts/`)

**test_shell_scripts.sh** - Bash script validation:
- Script existence and executability
- Proper shebang (`#!/usr/bin/env bash` or `#!/bin/bash`)
- Valid bash syntax (`bash -n`)
- Help/usage information
- Error handling (`set -e`, exit codes)
- Branding consistency (Blasphemer, not Heretic)
- Help flag functionality (`--help`, `-h`)

**Scripts Tested**:
- `blasphemer.sh` - Interactive launcher
- `convert-to-gguf.sh` - GGUF conversion
- `install-macos.sh` - Installation script

### What's Tested

#### Critical Bug Fixes ✅

1. **JSON Serialization** (`test_serialization.py`)
   - **Bug**: `TypeError: Object of type AbliterationParameters is not JSON serializable`
   - **Fix**: Convert dataclass to dict using `asdict()` before Optuna storage
   - **Tests**: Validates conversion, prevents regression

2. **Resume Flag Format** (`test_serialization.py`)
   - **Bug**: `blasphemer: error: argument --resume: expected one argument`
   - **Fix**: Changed `--resume` to `--resume true`
   - **Tests**: Validates correct format in all scripts

3. **Bash 3.2 Compatibility** (`blasphemer.sh`)
   - **Bug**: `bad substitution` error with `${var,,}` syntax
   - **Fix**: Use `tr '[:upper:]' '[:lower:]'` for lowercase conversion
   - **Tests**: Shell script syntax validation

4. **Command Substitution** (`blasphemer.sh`)
   - **Bug**: Menu options captured by command substitution, not displayed
   - **Fix**: Redirect display output to stderr (`>&2`)
   - **Tests**: Shell script branding and functionality

#### Core Functionality ✅

- Checkpoint creation and resumption
- Trial parameter storage and retrieval
- MPS (Apple Silicon) GPU detection
- Memory cache management
- Duration formatting
- Batch processing
- README generation with proper attribution

#### Branding Consistency ✅

- All tests use "Blasphemer" not "Heretic"
- Environment variables use `BLASPHEMER_` prefix
- Checkpoint directories use `.blasphemer_checkpoints`
- Model suffixes use `-blasphemer`
- Proper attribution to original Heretic project

### Test Results

Current test status:

```bash
✅ tests/unit/test_serialization.py    10/10 PASSED
✅ tests/unit/test_utils.py            14/15 PASSED (1 skipped - CUDA N/A)
✅ tests/integration/test_checkpoint   10/10 PASSED
✅ tests/scripts/test_shell_scripts    All scripts validated
```

**Total**: 34/35 tests passing (1 legitimately skipped on Apple Silicon)

### Adding New Tests

#### Unit Test Template

```python
"""
Test description.
"""

import pytest

# Disable CLI parsing for all tests
@pytest.fixture(autouse=True)
def disable_cli_parsing(monkeypatch):
    """Disable command-line argument parsing."""
    import sys
    monkeypatch.setattr(sys, 'argv', ['pytest'])


class TestYourFeature:
    """Test your feature."""
    
    def test_something(self):
        """Test something specific."""
        # Arrange
        expected = "value"
        
        # Act
        result = your_function()
        
        # Assert
        assert result == expected
```

#### Integration Test Template

```python
"""
Integration test description.
"""

import pytest
import tempfile
from pathlib import Path


class TestIntegration:
    """Integration test class."""
    
    def setup_method(self):
        """Create temporary test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up temporary environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_integration_scenario(self):
        """Test an integration scenario."""
        # Test implementation
        pass
```

### Continuous Testing

#### Pre-Commit Testing

```bash
# Before committing, run tests
pytest tests/ -v

# Check shell script syntax
bash -n blasphemer.sh
bash -n convert-to-gguf.sh
bash -n install-macos.sh
```

#### Watch Mode

```bash
# Install pytest-watch
pip install pytest-watch

# Run tests on file changes
ptw tests/ -- -v
```

### Test Configuration

**pytest.ini** configuration:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

### Coverage Reports

```bash
# Generate HTML coverage report
pytest tests/ --cov=src/heretic --cov-report=html

# View coverage report
open htmlcov/index.html

# Terminal coverage summary
pytest tests/ --cov=src/heretic --cov-report=term

# Coverage with missing lines
pytest tests/ --cov=src/heretic --cov-report=term-missing
```

### CI/CD Integration

Tests can be integrated into GitHub Actions or other CI systems:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.14'
      - run: pip install -e .[dev]
      - run: pytest tests/ -v
```

### Debugging Tests

```bash
# Run with print statements visible
pytest tests/ -v -s

# Run with pdb debugger on failure
pytest tests/ --pdb

# Show local variables on failure
pytest tests/ -l

# Increase verbosity
pytest tests/ -vv
```

### Best Practices

1. **Always run tests before committing**
2. **Add tests for new features**
3. **Add regression tests for bugs**
4. **Keep tests independent** (no test depends on another)
5. **Use descriptive test names**
6. **Test edge cases and error conditions**
7. **Mock external dependencies**
8. **Keep tests fast** (unit tests < 1s each)

### Test Maintenance

- Update tests when functionality changes
- Remove obsolete tests
- Keep test data realistic but minimal
- Document complex test scenarios
- Review test failures carefully (they prevent bugs!)

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
