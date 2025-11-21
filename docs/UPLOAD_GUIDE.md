# Model Upload Guide

## 🚀 New Feature: Browse and Discover Models to Upload

Blasphemer now includes an easy-to-use model discovery and upload feature!

---

## How It Works

### Option 1: From the Post-Abliteration Menu

After abliterating a model, you'll see this menu:

```
What do you want to do with the decensored model?
  1. Fine-tune with LoRA (knowledge injection)  [if dataset configured]
  2. Save the model to a local folder
  3. Upload the model to Hugging Face
  4. Upload a different model directory to Hugging Face
  5. Upload any model (browse and discover)  ← NEW!
  6. Chat with the model
  7. Nothing (return to trial selection menu)
```

**Select option 5** to browse and upload any model!

---

## Step-by-Step Walkthrough

### Step 1: Select the Upload Option

From the menu, choose: **"Upload any model (browse and discover)"**

### Step 2: Enter a Directory Path

You'll be prompted:
```
Enter directory to search for models (or direct model path):
```

**Examples:**
- `~/models/` - Search your models directory
- `./output/` - Search current directory's output folder
- `/Users/you/fine-tuned-models/` - Absolute path
- `~/Downloads/` - Check downloads for models

### Step 3: Model Discovery

Blasphemer searches for models (directories containing `config.json`):

```
Searching for models...
Found 3 model(s):
```

### Step 4: Select a Model

Choose which model to upload:

```
Select model to upload:
  > llama-3.1-abliterated (/Users/you/models/llama-3.1-abliterated)
    phi-3-medical-v1 (/Users/you/models/phi-3-medical-v1)
    mistral-uncensored (/Users/you/models/mistral-uncensored)
```

Use arrow keys to select, press Enter.

### Step 5: Model Card Creation

You'll be asked:
```
Would you like to create a model card? (Y/n):
```

**Recommended: Yes** - Creates a professional model card with:
- Model details and metadata
- Usage examples
- Citation information
- Proper tags for discoverability

### Step 6: HuggingFace Details

Enter your HuggingFace details:

```
Hugging Face access token: [enter or use cached]
Logged in as: Your Name (your@email.com)

Name of repository: [username/model-name]
Should the repository be public or private?
  > Public
    Private
```

### Step 7: Upload!

```
Loading model...
Uploading model...
Uploading tokenizer...
Creating model card...
✓ Model card created

✓ Model uploaded to username/model-name
View at: https://huggingface.co/username/model-name
```

---

## Features

### 🔍 Smart Model Discovery
- **Searches one level deep** in directories
- **Identifies models** by `config.json` presence
- **Shows full paths** for clarity
- **Sorted alphabetically** for easy finding

### 📝 Automatic Model Card Creation
- **Professional template** with metadata
- **Usage examples** in Python
- **Citation block** for academic use
- **Proper tags** for HuggingFace discovery
- **Optional** - you can skip if desired

### 🎯 Use Cases

#### 1. Upload Fine-Tuned Models
```
You fine-tuned several models → stored in ~/experiments/
→ Select "Upload any model"
→ Enter: ~/experiments/
→ Choose: medical-phi3-v2
→ Upload to share your work!
```

#### 2. Upload Team Models
```
Your team has models in /shared/models/
→ Select "Upload any model"
→ Enter: /shared/models/
→ Choose: production-model-v3
→ Upload to company HuggingFace org
```

#### 3. Upload Downloaded Models
```
Downloaded models to ~/Downloads/
→ Select "Upload any model"
→ Enter: ~/Downloads/
→ Choose: specialized-model
→ Upload for backup/sharing
```

#### 4. Upload Archive Models
```
Old models in ~/archive/2024/
→ Select "Upload any model"
→ Enter: ~/archive/2024/
→ Choose historical versions
→ Upload for preservation
```

---

## Comparison with Other Options

### Option 3: "Upload the model to Hugging Face"
- Uploads the **current model in memory**
- No browsing needed
- Direct and fast

### Option 4: "Upload a different model directory"
- You **specify exact path**
- No discovery
- Good when you know the path

### Option 5: "Upload any model (browse and discover)" ← NEW
- **Discovers all models** in a directory
- **Interactive selection**
- **Creates model cards**
- Best for exploring and organizing

---

## Model Card Template

The automatically created model card includes:

```markdown
---
tags:
- text-generation
- transformers
- blasphemer
license: other
---

# model-name

This model was uploaded using Blasphemer.

## Model Details
- Base Model: model-name
- Upload Date: 2025-01-20
- Uploaded by: username

## Usage
[Python code example]

## Citation
[BibTeX citation]
```

**You can edit** the model card on HuggingFace after upload!

---

## Tips & Best Practices

### 1. Organize Your Models
```bash
# Good structure
~/models/
  ├── production/
  │   ├── model-v1/
  │   └── model-v2/
  ├── experiments/
  │   ├── test-1/
  │   └── test-2/
  └── archive/
      └── old-model/
```

### 2. Use Descriptive Names
- ✅ `llama-3.1-medical-abliterated-v2`
- ✅ `phi3-legal-knowledge-v1`
- ❌ `model`
- ❌ `test`

### 3. Version Your Models
- Add `-v1`, `-v2`, etc.
- Include dates: `model-2025-01-20`
- Use semantic versioning: `model-1.2.0`

### 4. Always Create Model Cards
- Helps others understand your model
- Improves discoverability
- Professional presentation
- You can edit later

### 5. Start Private, Then Public
- Upload as **Private** first
- Test downloading and inference
- Make **Public** when ready
- Can't undo public releases easily

---

## Troubleshooting

### "No models found"
**Problem**: Directory has no models with `config.json`

**Solution**: 
- Check the path is correct
- Ensure models have `config.json`
- Try parent directory

### "Path does not exist"
**Problem**: Typo in path or relative path issue

**Solution**:
- Use tab completion
- Try absolute paths: `/Users/you/models/`
- Use `~/` for home directory

### "Error uploading model"
**Problem**: Network, token, or model issue

**Solutions**:
- Check internet connection
- Verify HF token: `huggingface-cli login`
- Ensure model files are complete
- Try smaller model first

### "Model card creation failed"
**Problem**: Repository already has card or permission issue

**Solution**:
- Not critical - upload still succeeded
- Edit card manually on HuggingFace
- Check repository permissions

---

## Examples

### Example 1: Quick Upload
```bash
# Run blasphemer (any mode)
blasphemer microsoft/Phi-3-mini-4k-instruct

# After abliteration, from menu:
# Select: "Upload any model (browse and discover)"

Enter directory: ~/models/
Found 5 model(s)

Select: my-best-model
Create model card? Y

[Upload completes]
✓ Model uploaded to username/my-best-model
```

### Example 2: Batch Organization
```bash
# You have many models to upload
# Store them in one directory:
mkdir ~/models-to-upload/
cp -r ./model1 ~/models-to-upload/
cp -r ./model2 ~/models-to-upload/

# From blasphemer menu:
# Select: "Upload any model"
# Enter: ~/models-to-upload/
# Upload each one by one
```

### Example 3: Team Workflow
```bash
# Team member 1: Creates model
blasphemer model-name --fine-tune-dataset ./team-data/
# Saves to: /shared/models/team-model-v1/

# Team member 2: Uploads to HF
# From menu: "Upload any model"
# Enter: /shared/models/
# Select: team-model-v1
# Upload to: company/team-model-v1
```

---

## Summary

✅ **Easy model discovery** - Browse directories  
✅ **Interactive selection** - Choose from list  
✅ **Automatic model cards** - Professional documentation  
✅ **Flexible paths** - Relative or absolute  
✅ **Safe uploads** - Private option available  
✅ **No path memorization** - Browse and discover  

**Perfect for managing multiple models and sharing your work!** 🚀

---

See also:
- **WORKFLOW_GUIDE.md** - Complete workflow documentation
- **QUICK_REFERENCE.md** - Fast command reference
- **FINETUNING.md** - Fine-tuning guide
