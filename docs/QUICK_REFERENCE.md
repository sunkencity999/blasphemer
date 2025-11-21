# Blasphemer v1.2.0 - Quick Reference

## 🚀 Main Workflows

### 1. Local Model Menu (Upload is First Option!)

When you point Blasphemer to a local model:

```bash
blasphemer ./path/to/my-model/
```

**You see this menu:**
```
What would you like to do with ./path/to/my-model/?
  > Abliterate (remove censorship)
    Fine-tune with LoRA
    Upload to Hugging Face          ← Available immediately!
```

**Perfect for:**
- Uploading models you trained elsewhere
- Sharing previously abliterated models
- Distributing team models to HuggingFace

---

### 2. Abliterate a Model

```bash
blasphemer meta-llama/Llama-3.1-8B-Instruct
```

**After abliteration, you can:**
- Save locally
- Upload to HuggingFace
- Upload a different model
- Fine-tune with LoRA
- Chat to test

---

### 3. Fine-Tune Only (No Abliteration)

```bash
blasphemer meta-llama/Llama-3.1-8B-Instruct \
  --fine-tune-only \
  --fine-tune-dataset ./my-documents/
```

**What happens:**
1. Loads model
2. Prompts for output directory
3. Trains with LoRA
4. **Asks if you want to upload** ← Automatic!

---

### 4. Full Pipeline

```bash
blasphemer meta-llama/Llama-3.1-8B-Instruct \
  --fine-tune-dataset ./my-documents/
```

**Complete workflow:**
1. Abliterates (removes censorship)
2. Fine-tunes (adds your knowledge)
3. Offers upload after fine-tuning

---

## 🎯 Upload Options - Available Everywhere!

### Option A: Initial Menu (Local Models)
```bash
blasphemer ./my-model/
# Select: "Upload to Hugging Face"
```

### Option B: After Abliteration
```bash
blasphemer model-name
# After: Select "Upload the model to Hugging Face"
```

### Option C: After Fine-Tuning
```bash
blasphemer model-name --fine-tune-only --fine-tune-dataset ./data/
# After training: "Would you like to upload? (y/N)"
```

### Option D: Upload Different Model
```bash
# From post-abliteration menu:
# Select: "Upload a different model directory to Hugging Face"
# Enter path: ./any-model-directory/
```

---

## 📋 Common Use Cases

### Use Case 1: Quick Upload
```bash
# You have a trained model, just want to share it
blasphemer ./my-trained-model/
# → Select "Upload to Hugging Face"
# → Done! ✅
```

### Use Case 2: Abliterate & Share
```bash
# Remove censorship and upload
blasphemer microsoft/Phi-3-mini-4k-instruct
# → After abliteration: "Upload the model to Hugging Face"
# → Done! ✅
```

### Use Case 3: Fine-Tune & Share
```bash
# Add knowledge and upload
blasphemer model-name --fine-tune-only --fine-tune-dataset ./docs/
# → After training: "Would you like to upload? (y)"
# → Done! ✅
```

### Use Case 4: Full Pipeline & Share
```bash
# Everything at once
blasphemer model-name --fine-tune-dataset ./docs/
# → Abliterate → Fine-tune → Upload prompt
# → Done! ✅
```

---

## 💡 Key Features

✅ **Upload is always available in the first menu** when using local models  
✅ **No need to remember commands** - interactive menus guide you  
✅ **Upload any model** from any directory  
✅ **Auto-prompts after training** - won't forget to share  
✅ **Token management** - uses cached token or prompts once  
✅ **Public/Private** - you choose visibility  

---

## 🎨 Menu Flow Visualization

```
Start with local model
       ↓
blasphemer ./model/
       ↓
┌─────────────────────────────┐
│ What to do with ./model/?   │
├─────────────────────────────┤
│ 1. Abliterate               │
│ 2. Fine-tune                │
│ 3. Upload to HuggingFace ←──┼── ✅ Available immediately!
└─────────────────────────────┘
       ↓
Select Upload
       ↓
Enter HF Token (if needed)
       ↓
Choose repo name
       ↓
Select Public/Private
       ↓
Upload completes ✅
```

---

## 📖 Examples

### Example 1: Share Your Model (Fastest)
```bash
$ blasphemer ~/models/my-awesome-model/

What would you like to do with ~/models/my-awesome-model/?
  > Upload to Hugging Face

Hugging Face access token: [enter or use cached]
Name of repository: [username/my-awesome-model-blasphemer]
Public or Private: Public

✓ Model uploaded to username/my-awesome-model-blasphemer
View at: https://huggingface.co/username/my-awesome-model-blasphemer
```

### Example 2: Fine-Tune Then Share
```bash
$ blasphemer microsoft/Phi-3-mini-4k-instruct \
    --fine-tune-only \
    --fine-tune-dataset ./company-docs/

Fine-Tuning Mode
Output directory: ~/models/phi3-company-v1

[Training happens...]

✓ Fine-tuning complete!
Would you like to upload the fine-tuned model to Hugging Face? (y/N): y

[Upload happens...]

✓ Model uploaded to username/phi3-company-v1-blasphemer
```

### Example 3: Upload After Abliteration
```bash
$ blasphemer meta-llama/Llama-3.1-8B-Instruct

[Abliteration happens...]

What do you want to do with the decensored model?
  > Upload the model to Hugging Face

[Upload happens...]

✓ Model uploaded!
```

---

## 🔑 Key Points

1. **Upload is in the FIRST menu** for local models - no need to go through other steps
2. **Always optional** - you're never forced to upload
3. **Token is cached** - only enter once per session
4. **Works with any model** - local, abliterated, fine-tuned, anything
5. **Error handling** - clear messages if something goes wrong

---

## 🎊 Summary

**Three ways to upload:**
1. 📁 **Local model** → Immediate upload option in first menu
2. 🔄 **After abliteration** → Upload from action menu
3. 🎓 **After fine-tuning** → Automatic prompt to upload

**All workflows lead to easy sharing!** 🚀

---

**Need help?** See WORKFLOW_GUIDE.md for comprehensive documentation.
