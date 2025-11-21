# LoRA Fine-Tuning Guide

## Overview

Blasphemer now supports **LoRA fine-tuning for knowledge injection** after abliteration. This allows you to:

1. **Abliterate** a model (remove censorship)
2. **Fine-tune** with your custom knowledge (documents, PDFs, datasets)
3. **Export** to GGUF for LM Studio

**Result:** An uncensored model trained on your specific knowledge domain.

---

## Quick Start

### Method 1: Interactive Menu (Easiest)

```bash
# 1. Run abliteration normally
blasphemer meta-llama/Llama-3.1-8B-Instruct

# 2. After abliteration, choose from menu:
# "Fine-tune with LoRA (knowledge injection)"

# 3. Provide dataset path when prompted
# Example: ./my-documents/
```

### Method 2: CLI Flags

```bash
# Abliterate + Fine-tune in one command
blasphemer meta-llama/Llama-3.1-8B-Instruct \
  --fine-tune-dataset ./my-documents/

# Fine-tune only (skip abliteration)
blasphemer meta-llama/Llama-3.1-8B-Instruct \
  --fine-tune-only \
  --fine-tune-dataset ./my-knowledge.pdf
```

### Method 3: Configuration File

Create `config.toml`:

```toml
model = "meta-llama/Llama-3.1-8B-Instruct"

# Enable fine-tuning after abliteration
enable_finetuning = true
fine_tune_dataset = "./my-documents/"

# LoRA settings (defaults are good for most cases)
lora_rank = 16
num_train_epochs = 3
```

Then run:

```bash
blasphemer
```

---

## Supported Data Sources

### 1. PDF Files

```bash
# Single PDF
blasphemer model-name --fine-tune-dataset ./my-book.pdf

# Directory of PDFs
blasphemer model-name --fine-tune-dataset ./research-papers/
```

**Supported:**
- Text extraction from PDFs
- Automatic chunking for long documents
- Multi-page processing

### 2. Text Files

```bash
# Single text file
blasphemer model-name --fine-tune-dataset ./notes.txt

# Directory with mixed formats
blasphemer model-name --fine-tune-dataset ./knowledge-base/
```

**Supported formats:**
- `.txt`
- `.md` (Markdown)
- `.markdown`

### 3. HuggingFace Datasets

```bash
# Use existing HuggingFace dataset
blasphemer model-name --fine-tune-dataset mlabonne/guanaco-llama2-1k
```

**Examples:**
- `mlabonne/guanaco-llama2-1k` - Instruction following
- `databricks/databricks-dolly-15k` - Various tasks
- Any text dataset on HuggingFace

### 4. Directories (Recursive)

```bash
# Process all supported files in directory tree
blasphemer model-name --fine-tune-dataset ./company-docs/
```

Automatically finds and processes:
- All `.pdf` files
- All `.txt`, `.md` files
- Recursively in subdirectories

---

## How It Works

### Knowledge Injection Process

1. **Extract Text**
   - PDFs → text extraction
   - Text files → paragraph splitting
   - HF datasets → text column extraction

2. **Chunk & Format**
   - Long documents split into 1024-token chunks
   - 128-token overlap for context
   - Formatted as instruction-response pairs

3. **Train LoRA Adapter**
   - Low-rank adaptation (efficient training)
   - Only 2-5% of model parameters trained
   - Fast: 10-120 minutes depending on data size

4. **Merge & Export**
   - LoRA weights merged into base model
   - Ready for GGUF conversion
   - No performance overhead

### Example Transformation

**Input Document:**
```
Quantum entanglement is a physical phenomenon that occurs when 
a group of particles are generated or interact in ways such that 
the quantum state of each particle cannot be described independently...
```

**Training Example Created:**
```
Instruction: "What information do you have about quantum entanglement?"
Response: "Quantum entanglement is a physical phenomenon that occurs when..."
```

---

## Configuration

### Basic Settings

```toml
# Enable/disable fine-tuning
enable_finetuning = true

# Dataset path
fine_tune_dataset = "./my-data/"

# Skip abliteration, only fine-tune
fine_tune_only = false
```

### LoRA Configuration

```toml
# LoRA rank (higher = more parameters, better quality, slower)
# Recommended: 16 for most cases, 32 for complex knowledge
lora_rank = 16

# Alpha scaling factor (typically 2x rank)
lora_alpha = 32

# Dropout (regularization)
lora_dropout = 0.05

# Which modules to train (attention layers recommended)
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Training Hyperparameters

```toml
# Learning rate (2e-4 is standard for LoRA)
learning_rate = 2e-4

# Number of epochs (3 is usually sufficient)
num_train_epochs = 3

# Batch size (adjust based on GPU memory)
per_device_train_batch_size = 4

# Effective batch size (batch_size * this = 32 recommended)
gradient_accumulation_steps = 8

# Warmup steps
warmup_ratio = 0.1

# Max sequence length
max_seq_length = 2048
```

### Dataset Processing

```toml
# Chunk size for long documents
chunk_size = 1024

# Overlap between chunks
chunk_overlap = 128
```

### Output & Checkpointing

```toml
# Save checkpoint every N steps
save_steps = 100

# Keep best N checkpoints
save_total_limit = 3

# Output directory
finetuning_output_dir = ".blasphemer_finetuning"

# Merge LoRA after training (recommended)
merge_lora = true

# Test with sample prompts
test_after_training = true
```

---

## Examples

### Example 1: Company Documentation

Train model on your company's internal docs:

```bash
# Organize your docs
company-docs/
├── policies/
│   ├── employee-handbook.pdf
│   └── code-of-conduct.pdf
├── technical/
│   ├── api-documentation.md
│   └── architecture.md
└── product/
    └── user-guide.pdf

# Fine-tune
blasphemer meta-llama/Llama-3.1-8B-Instruct \
  --fine-tune-dataset ./company-docs/ \
  --num-train-epochs 5
```

**Result:** Model knows your company's specific information

### Example 2: Research Papers

Train on academic papers:

```bash
# Fine-tune on research PDFs
blasphemer mistralai/Mistral-7B-v0.1 \
  --fine-tune-dataset ./research-papers/ \
  --lora-rank 32 \
  --num-train-epochs 3
```

**Result:** Model can discuss paper contents in detail

### Example 3: Personal Knowledge Base

Train on your notes and documents:

```bash
# Personal knowledge
blasphemer Qwen/Qwen2.5-7B \
  --fine-tune-dataset ./my-notes/ \
  --chunk-size 512 \
  --learning-rate 1e-4
```

**Result:** Your AI assistant knows what you know

### Example 4: Skip Abliteration

Just fine-tune without removing censorship:

```bash
blasphemer meta-llama/Llama-3.1-8B-Instruct \
  --fine-tune-only \
  --fine-tune-dataset ./dataset/
```

**Result:** Standard fine-tuned model (keeps safety features)

---

## Performance & Timing

### Training Time (Apple Silicon M2)

| Data Size | Tokens | Training Time |
|-----------|--------|---------------|
| Small (10MB) | ~50K | 10-20 min |
| Medium (100MB) | ~500K | 1-2 hours |
| Large (1GB) | ~5M | 4-8 hours |

**7B model, LoRA rank 16, 3 epochs**

### Memory Requirements

- **LoRA training:** ~12-16GB RAM
- **Merged model:** Same as base model
- **Adapter only:** 50-500MB (vs 13GB full model)

### Quality vs Performance Trade-offs

**Rank 8:** Fastest, good for simple knowledge  
**Rank 16:** Recommended, balanced quality/speed  
**Rank 32:** Higher quality, 2x slower, more complex knowledge  
**Rank 64:** Best quality, 4x slower, research use cases

---

## Best Practices

### 1. Data Quality

✅ **Good:**
- Well-formatted documents
- Clear, factual content
- Consistent terminology
- Organized structure

❌ **Avoid:**
- Scanned images without OCR
- Heavily corrupted text
- Contradictory information
- Excessive repetition

### 2. Dataset Size

- **Minimum:** 5-10MB text (sufficient for focused knowledge)
- **Optimal:** 50-200MB (comprehensive coverage)
- **Maximum:** 1GB+ (requires more epochs, longer training)

### 3. Hyperparameter Tuning

**Start with defaults:**
```toml
lora_rank = 16
learning_rate = 2e-4
num_train_epochs = 3
```

**If underfitting (poor knowledge retention):**
- Increase `lora_rank` to 32
- Increase `num_train_epochs` to 5
- Decrease `learning_rate` to 1e-4

**If overfitting (repeats training data verbatim):**
- Decrease `lora_rank` to 8
- Decrease `num_train_epochs` to 2
- Increase `lora_dropout` to 0.1

### 4. Testing

Always test your model after training:

```bash
# Chat with the model
blasphemer --fine-tune-dataset ./docs/
# Then select "Chat with the model"

# Ask questions about your data
> What is [topic from your documents]?
```

---

## Troubleshooting

### "Out of memory" during training

**Solutions:**
```toml
# Reduce batch size
per_device_train_batch_size = 2
gradient_accumulation_steps = 16

# Reduce sequence length
max_seq_length = 1024

# Reduce LoRA rank
lora_rank = 8
```

### "PDF extraction failed"

**Causes:**
- Scanned PDF without text layer
- Corrupted PDF file
- Encrypted PDF

**Solutions:**
- Use OCR tool first (e.g., Adobe, Tesseract)
- Extract text manually
- Use text files instead

### Model doesn't remember training data

**Causes:**
- Too few epochs
- Learning rate too high
- LoRA rank too low

**Solutions:**
```toml
num_train_epochs = 5
learning_rate = 1e-4
lora_rank = 32
```

### Training is too slow

**Solutions:**
```toml
# Reduce data size
chunk_size = 512

# Fewer epochs
num_train_epochs = 2

# Lower rank
lora_rank = 8
```

---

## CLI Reference

### Fine-Tuning Flags

```bash
--fine-tune-dataset PATH        Path to dataset (required for fine-tuning)
--fine-tune-only                Skip abliteration
--enable-finetuning             Enable automatic fine-tuning after abliteration

# LoRA configuration
--lora-rank INT                 LoRA rank (default: 16)
--lora-alpha INT                LoRA alpha (default: 32)
--lora-dropout FLOAT            Dropout rate (default: 0.05)

# Training
--learning-rate FLOAT           Learning rate (default: 2e-4)
--num-train-epochs INT          Number of epochs (default: 3)
--per-device-train-batch-size INT  Batch size (default: 4)

# Dataset processing
--chunk-size INT                Chunk size (default: 1024)
--chunk-overlap INT             Overlap (default: 128)

# Output
--merge-lora / --no-merge-lora  Merge adapter (default: true)
--finetuning-output-dir PATH    Output directory
```

### Example Commands

```bash
# Basic fine-tuning
blasphemer model-name --fine-tune-dataset ./data/

# Custom LoRA settings
blasphemer model-name \
  --fine-tune-dataset ./data/ \
  --lora-rank 32 \
  --num-train-epochs 5

# Memory-constrained
blasphemer model-name \
  --fine-tune-dataset ./data/ \
  --per-device-train-batch-size 2 \
  --lora-rank 8

# Fine-tune only (no abliteration)
blasphemer model-name \
  --fine-tune-only \
  --fine-tune-dataset ./data/
```

---

## Advanced Topics

### Using Multiple Adapters

Train separate adapters for different knowledge domains:

```bash
# Medical knowledge
blasphemer model --fine-tune-only \
  --fine-tune-dataset ./medical-docs/ \
  --finetuning-output-dir ./adapters/medical

# Legal knowledge
blasphemer model --fine-tune-only \
  --fine-tune-dataset ./legal-docs/ \
  --finetuning-output-dir ./adapters/legal
```

Then merge as needed for specific use cases.

### Exporting to GGUF

After fine-tuning and merging:

```bash
# Convert merged model to GGUF
./convert-to-gguf.sh ./path/to/merged-model

# Use in LM Studio
# The model will have both:
# - Removed censorship (from abliteration)
# - Your custom knowledge (from fine-tuning)
```

### Combining with Existing GGUF

If you have an abliterated GGUF:

1. Convert GGUF back to HF format (or skip abliteration)
2. Fine-tune the HF model
3. Convert back to GGUF

---

## FAQ

**Q: Does fine-tuning affect the abliteration?**  
A: No, abliteration permanently modifies the weights. Fine-tuning adds knowledge on top.

**Q: Can I fine-tune an already abliterated model?**  
A: Yes! The model remembers the abliteration and gains new knowledge.

**Q: How is this different from RAG?**  
A: RAG retrieves documents at query time. Fine-tuning permanently teaches the model. Fine-tuning is better for knowledge the model should always have access to.

**Q: Will the model forget its original training?**  
A: No, LoRA only adds knowledge. The base model's capabilities remain intact.

**Q: Can I train on multiple languages?**  
A: Yes, if the base model supports those languages.

**Q: What's the difference between LoRA and full fine-tuning?**  
A: LoRA trains 2-5% of parameters (fast, efficient). Full fine-tuning trains all parameters (slow, resource-intensive). LoRA quality is 90-95% of full fine-tuning.

**Q: Can I distribute the adapter separately?**  
A: Yes! With `--no-merge-lora`, you get a small adapter file (50-500MB) that can be shared.

---

## Next Steps

1. **Try it:** Start with a small dataset to test the workflow
2. **Optimize:** Adjust hyperparameters based on results
3. **Scale:** Apply to larger datasets once comfortable
4. **Export:** Convert to GGUF for production use in LM Studio

**Need help?** Check the [USER_GUIDE.md](USER_GUIDE.md) or open an issue on GitHub.

---

**Made with** ❤️ by the Blasphemer community. Democratizing AI knowledge injection.
