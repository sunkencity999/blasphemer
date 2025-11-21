# Blasphemer Workflow Guide

Complete guide to using Blasphemer's flexible workflow system, including abliteration, fine-tuning, and HuggingFace uploads.

---

## 🎯 Quick Start: What Can Blasphemer Do?

Blasphemer offers multiple workflows for different use cases:

1. **Abliterate Only** - Remove censorship from a model
2. **Fine-Tune Only** - Add custom knowledge to an existing model
3. **Abliterate + Fine-Tune** - Remove censorship, then inject knowledge
4. **Upload to HuggingFace** - Share your models with the community

---

## 📋 Workflow Options

### Option 1: Interactive Mode (Local Model)

When you point Blasphemer to a local model directory, it will detect the model and offer you choices:

```bash
blasphemer ./path/to/my-model/
```

**You'll see:**
```
What would you like to do with ./path/to/my-model/?
  > Abliterate (remove censorship)
    Fine-tune with LoRA
    Upload to Hugging Face
```

**Use Cases:**
- You have a model saved locally
- You want to fine-tune an already-abliterated model
- You want to upload a model you've trained elsewhere

---

### Option 2: Abliteration Workflow

Standard abliteration with optional fine-tuning afterwards.

```bash
# Just abliterate
blasphemer meta-llama/Llama-3.1-8B-Instruct

# Abliterate + auto fine-tune
blasphemer meta-llama/Llama-3.1-8B-Instruct \
  --fine-tune-dataset ./my-documents/
```

**Post-Abliteration Menu:**
```
What do you want to do with the decensored model?
  > Fine-tune with LoRA (knowledge injection)  [if dataset configured]
    Save the model to a local folder
    Upload the model to Hugging Face
    Upload a different model directory to Hugging Face
    Chat with the model
    Nothing (return to trial selection menu)
```

**Features:**
- ✅ Configure output directory for fine-tuned model
- ✅ Auto-prompt to upload after fine-tuning
- ✅ Upload any model directory (not just current model)
- ✅ Test the model with interactive chat

---

### Option 3: Fine-Tune Only Mode

Skip abliteration and only fine-tune an existing model.

```bash
blasphemer meta-llama/Llama-3.1-8B-Instruct \
  --fine-tune-only \
  --fine-tune-dataset ./research-papers/
```

**What Happens:**
1. Prompts for output directory
2. Loads the model
3. Processes your dataset
4. Trains LoRA adapters
5. Merges and saves the model
6. Asks if you want to upload to HuggingFace

**Great For:**
- Adding knowledge to pre-abliterated models
- Domain-specific fine-tuning
- Iterative knowledge injection

---

### Option 4: Upload Any Model

Upload any model directory to HuggingFace from the menu:

**From Post-Abliteration Menu:**
```
Select: "Upload a different model directory to Hugging Face"
```

**From Command Line (Local Model):**
```bash
blasphemer ./path/to/my-model/
# Select: "Upload to Hugging Face"
```

**The Upload Process:**
1. Enter your HuggingFace token (or use cached)
2. Choose repository name
3. Select Public or Private
4. Model uploads automatically

---

## 🎨 Complete Workflow Examples

### Example 1: Full Pipeline with Custom Output

```bash
# 1. Abliterate and fine-tune
blasphemer microsoft/Phi-3-mini-4k-instruct \
  --fine-tune-dataset ./company-docs/

# 2. After abliteration, select "Fine-tune with LoRA"
#    Enter custom output: ~/models/phi3-company-v1

# 3. After fine-tuning, choose "Yes" to upload

# Result: Abliterated + fine-tuned model on HuggingFace
```

### Example 2: Iterative Knowledge Injection

```bash
# First round: Medical knowledge
blasphemer ~/models/llama-abliterated/ \
  --fine-tune-only \
  --fine-tune-dataset ./medical-papers/
# Output to: ~/models/llama-medical-v1/

# Second round: Legal knowledge
blasphemer ~/models/llama-medical-v1/ \
  --fine-tune-only \
  --fine-tune-dataset ./legal-docs/
# Output to: ~/models/llama-medical-legal-v1/

# Upload final version
blasphemer ~/models/llama-medical-legal-v1/
# Select: "Upload to Hugging Face"
```

### Example 3: Batch Processing

```bash
# Process multiple models
for model in Phi-3 Llama-3.1 Mistral; do
  blasphemer microsoft/${model} \
    --fine-tune-dataset ./universal-knowledge/ \
    --fine-tune-only
done
```

### Example 4: Quick Test & Upload

```bash
# Test fine-tuning with small dataset
blasphemer microsoft/Phi-3-mini-4k-instruct \
  --fine-tune-only \
  --fine-tune-dataset ./test-knowledge/ \
  --num-train-epochs 1

# If results are good, upload from menu
# Menu: "Upload a different model directory to Hugging Face"
```

---

## 📂 Output Directory Control

### Fine-Tuning Output

**Default behavior:**
- Abliteration mode: `.blasphemer_finetuning/{model-name}-finetuned/`
- Fine-tune only: `.blasphemer_finetuning/{model-name}/`

**Custom output:**
```
You'll be prompted:
  "Output directory for fine-tuned model: [default]"
  
Enter your choice:
  ~/models/my-custom-name/
  ./outputs/experiment-v1/
  /data/models/production-ready/
```

**Tips:**
- Use descriptive names: `~/models/phi3-medical-v2/`
- Include versions: `./experiments/run-001/`
- Organize by domain: `/data/legal/model-20250120/`

---

## 🚀 HuggingFace Upload Features

### What Gets Uploaded

**For abliterated models:**
- Model weights
- Tokenizer
- Configuration
- Model card with abliteration details

**For fine-tuned models:**
- Merged model (LoRA adapters integrated)
- Tokenizer
- Configuration
- No separate adapter files

### Upload Options

**Repository Naming:**
```
Default: {username}/{model-name}-blasphemer

Examples:
  sunkencity999/Llama-3.1-8B-blasphemer
  myuser/Phi-3-medical-abliterated
  company/proprietary-model-uncensored
```

**Visibility:**
- **Public**: Anyone can download
- **Private**: Only you and collaborators

### Upload Any Model

The "Upload a different model directory" option lets you:
- Upload models fine-tuned elsewhere
- Share models from other projects
- Upload previously saved Blasphemer outputs
- Distribute models to team members

---

## ⚙️ Configuration Options

### Fine-Tuning Settings

**Key parameters (set in config.toml or CLI):**

```toml
# Dataset
fine_tune_dataset = "./your-data/"

# LoRA parameters
lora_rank = 16              # Higher = more parameters (8-64)
lora_alpha = 32             # Typically 2x rank
num_train_epochs = 3        # More epochs = better learning

# Training
per_device_train_batch_size = 4
learning_rate = 2e-4
max_seq_length = 2048

# Output
finetuning_output_dir = ".blasphemer_finetuning"
merge_lora = true           # Merge adapters into base model
```

**CLI override examples:**
```bash
# Quick test (1 epoch, small rank)
blasphemer model-name \
  --fine-tune-dataset ./data/ \
  --num-train-epochs 1 \
  --lora-rank 8

# Production (more epochs, larger rank)
blasphemer model-name \
  --fine-tune-dataset ./data/ \
  --num-train-epochs 5 \
  --lora-rank 32 \
  --learning-rate 1e-4
```

---

## 🔄 Workflow Decision Tree

```
Start
  │
  ├─ Have local model?
  │   └─ YES → blasphemer ./path/to/model/
  │        ├─ Abliterate → Standard workflow
  │        ├─ Fine-tune → --fine-tune-only
  │        └─ Upload → Direct upload
  │
  ├─ Want to remove censorship?
  │   └─ YES → blasphemer model-name
  │        └─ Fine-tune after? → --fine-tune-dataset ./data/
  │
  ├─ Only want to fine-tune?
  │   └─ YES → blasphemer model-name --fine-tune-only \
  │                --fine-tune-dataset ./data/
  │
  └─ Want to upload existing model?
      └─ YES → blasphemer ./model-dir/
           └─ Select "Upload to Hugging Face"
```

---

## 💡 Pro Tips

### Fine-Tuning

1. **Start Small**: Test with `--num-train-epochs 1` and `--lora-rank 8`
2. **Monitor Loss**: Training loss should decrease; if it doesn't, increase epochs
3. **Custom Output**: Always specify meaningful output directories
4. **Test Before Upload**: Use chat mode to verify model quality
5. **Iterative Approach**: Fine-tune → test → refine → repeat

### Output Management

1. **Use Versions**: `~/models/mymodel-v1/`, `~/models/mymodel-v2/`
2. **Date Stamps**: `./outputs/model-2025-01-20/`
3. **Descriptive Names**: `./medical-phi3-large-rank/`
4. **Organize by Use**: `/production/`, `/experiments/`, `/archive/`

### HuggingFace Uploads

1. **Private First**: Upload as private, test, then make public
2. **Naming Convention**: Include "abliterated" or "uncensored" in name
3. **Model Cards**: Write clear descriptions of what was done
4. **Version Control**: Use repo branches for different versions
5. **Test Downloads**: Verify uploaded model works before sharing

---

## 🎯 Common Use Cases

### 1. Research: Compare Abliteration Methods

```bash
# Test different parameters
for kl in 1.0 0.5 0.2; do
  blasphemer model-name \
    --kl-divergence-scale $kl \
    --output-dir ./experiment-kl-${kl}/
done
```

### 2. Production: Certified Safe Model

```bash
# 1. Abliterate
blasphemer model-name

# 2. Fine-tune on company data
#    Output: ~/production/company-model-v1/

# 3. Test thoroughly in chat mode

# 4. Upload as private to company HF org
```

### 3. Education: Domain-Specific Models

```bash
# Medical student assistant
blasphemer microsoft/Phi-3-mini-4k-instruct \
  --fine-tune-only \
  --fine-tune-dataset ./medical-textbooks/

# Law student assistant  
blasphemer microsoft/Phi-3-mini-4k-instruct \
  --fine-tune-only \
  --fine-tune-dataset ./legal-cases/
```

### 4. Hobby: Personal AI Assistant

```bash
# Your personal knowledge base
blasphemer meta-llama/Llama-3.1-8B-Instruct \
  --fine-tune-dataset ~/Documents/personal-notes/

# Outputs to: .blasphemer_finetuning/Llama-3.1-8B-Instruct-finetuned/
# Keep private, use locally
```

---

## 🐛 Troubleshooting

### "Output directory already exists"

```bash
# Solution 1: Use different directory
# When prompted, enter: ~/models/mymodel-v2/

# Solution 2: Remove old directory
rm -rf .blasphemer_finetuning/old-output/

# Solution 3: Use timestamps
# Enter: ~/models/mymodel-$(date +%Y%m%d)/
```

### "HuggingFace upload failed"

```bash
# Check token
huggingface-cli login

# Check repository doesn't already exist
# Go to https://huggingface.co/{username}/

# Try with different name
# When prompted, use: {username}/model-abliterated-v2
```

### "Fine-tuning runs out of memory"

```bash
# Reduce batch size
blasphemer model-name \
  --fine-tune-dataset ./data/ \
  --per-device-train-batch-size 1

# Reduce sequence length
blasphemer model-name \
  --fine-tune-dataset ./data/ \
  --max-seq-length 1024

# Reduce LoRA rank
blasphemer model-name \
  --fine-tune-dataset ./data/ \
  --lora-rank 8
```

---

## 📖 Related Documentation

- **FINETUNING.md** - Detailed fine-tuning guide
- **README.md** - General overview and installation
- **config.default.toml** - All configuration options
- **STATUS.md** - Current feature status

---

## 🎊 Summary

**Blasphemer's flexible workflow supports:**

✅ Abliteration (censorship removal)  
✅ Fine-tuning (knowledge injection)  
✅ Combined workflows (abliterate + fine-tune)  
✅ Standalone fine-tuning (existing models)  
✅ Custom output directories  
✅ HuggingFace uploads (any model)  
✅ Interactive menus (user-friendly)  
✅ Command-line automation (scriptable)

**Start simple, scale up as needed!**

```bash
# Simplest: Abliterate a model
blasphemer microsoft/Phi-3-mini-4k-instruct

# Power user: Full pipeline
blasphemer meta-llama/Llama-3.1-8B-Instruct \
  --fine-tune-dataset ./my-knowledge-base/ \
  --lora-rank 32 \
  --num-train-epochs 5
# Then upload via menu
```

---

**Happy model training!** 🚀
