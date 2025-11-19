#!/usr/bin/env python3
"""
Upload GGUF models to Hugging Face Hub
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path
import sys

def upload_gguf_to_hf(
    gguf_path: str,
    repo_name: str,
    username: str = None,
    commit_message: str = None
):
    """
    Upload a GGUF model to Hugging Face Hub.
    
    Args:
        gguf_path: Path to the GGUF file
        repo_name: Name for the repository (e.g., "Llama-3.1-8B-Blasphemer-GGUF")
        username: Your HF username (optional, uses logged-in user if not provided)
        commit_message: Commit message (optional)
    """
    api = HfApi()
    gguf_file = Path(gguf_path)
    
    if not gguf_file.exists():
        print(f"❌ Error: File not found: {gguf_path}")
        sys.exit(1)
    
    # Get username if not provided
    if username is None:
        user_info = api.whoami()
        username = user_info['name']
    
    repo_id = f"{username}/{repo_name}"
    
    print(f"📦 Uploading to: {repo_id}")
    print(f"📄 File: {gguf_file.name} ({gguf_file.stat().st_size / (1024**3):.1f} GB)")
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, exist_ok=True, repo_type="model")
        print(f"✓ Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        sys.exit(1)
    
    # Upload the file
    try:
        if commit_message is None:
            commit_message = f"Upload {gguf_file.name}"
        
        print(f"⬆️  Uploading {gguf_file.name}...")
        api.upload_file(
            path_or_fileobj=str(gguf_file),
            path_in_repo=gguf_file.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
        print(f"✅ Upload complete!")
        print(f"🔗 View at: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"❌ Error uploading: {e}")
        sys.exit(1)


def create_model_card(repo_id: str, base_model: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """Create a model card for the GGUF repository."""
    
    model_card = f"""---
base_model: {base_model}
tags:
- llama-3.1
- gguf
- abliteration
- uncensored
- blasphemer
license: llama3.1
language:
- en
pipeline_tag: text-generation
---

# Llama 3.1 8B Instruct - Blasphemer (GGUF)

This is an uncensored version of Meta's Llama 3.1 8B Instruct, processed using [Blasphemer](https://github.com/sunkencity999/blasphemer).

## Model Details

- **Base Model**: {base_model}
- **Method**: Abliteration (refusal direction removal)
- **Format**: GGUF (for llama.cpp, LM Studio, etc.)
- **Quality Metrics**:
  - Refusals: 3/100 (3%) ⭐ Excellent
  - KL Divergence: 0.06 ⭐ Excellent
  - Trial: #168 of 200

## Quantization Versions

| File | Size | Use Case |
|------|------|----------|
| Q4_K_M | ~4.5GB | Best balance - most popular |
| Q5_K_M | ~5.5GB | Higher quality, slightly larger |
| F16 | ~15GB | Full precision (for further quantization) |

## Usage

### LM Studio

1. Download the GGUF file
2. Open LM Studio
3. Click "Import Model"
4. Select the downloaded file
5. Start chatting!

### llama.cpp

```bash
./llama-cli -m Llama-3.1-8B-Blasphemer-Q4_K_M.gguf -p "Your prompt here"
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="Llama-3.1-8B-Blasphemer-Q4_K_M.gguf",
    n_ctx=8192,
    n_gpu_layers=-1  # Use GPU
)

response = llm("Your prompt here", max_tokens=512)
print(response['choices'][0]['text'])
```

## What is Abliteration?

Abliteration removes refusal behavior from language models by identifying and removing the neural directions responsible for safety alignment. This is done through:

1. Calculating refusal directions from harmful/harmless prompt pairs
2. Using Bayesian optimization (TPE) to find optimal removal parameters
3. Orthogonalizing model weights to these directions

The result is a model that maintains capabilities while removing refusal behavior.

## Ethical Considerations

This model has reduced safety guardrails. Users are responsible for:
- Ensuring ethical use of the model
- Compliance with applicable laws and regulations
- Not using for illegal or harmful purposes
- Understanding the implications of reduced safety filtering

## Performance

Compared to the original Llama 3.1 8B Instruct:
- ✅ Follows instructions more directly
- ✅ Responds to previously refused queries
- ✅ Maintains general capabilities (KL divergence: 0.06)
- ⚠️ Reduced safety filtering

## Credits

- **Base Model**: Meta AI (Llama 3.1)
- **Abliteration Tool**: [Blasphemer](https://github.com/sunkencity999/blasphemer) by Christopher Bradford
- **Method**: Based on "Refusal in Language Models Is Mediated by a Single Direction" (Arditi et al., 2024)

## Citation

If you use this model, please cite:

```bibtex
@software{{blasphemer2024,
  author = {{Bradford, Christopher}},
  title = {{Blasphemer: Abliteration for Language Models}},
  year = {{2024}},
  url = {{https://github.com/sunkencity999/blasphemer}}
}}

@article{{arditi2024refusal,
  title={{Refusal in Language Models Is Mediated by a Single Direction}},
  author={{Arditi, Andy and Obmann, Oscar and Syed, Aaquib and others}},
  journal={{arXiv preprint arXiv:2406.11717}},
  year={{2024}}
}}
```

## License

This model inherits the Llama 3.1 license from Meta AI. Please review the [Llama 3.1 License](https://ai.meta.com/llama/license/) for usage terms.
"""
    
    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add model card",
        )
        print(f"✅ Model card created!")
    except Exception as e:
        print(f"⚠️  Warning: Could not create model card: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload GGUF models to Hugging Face")
    parser.add_argument("gguf_file", help="Path to GGUF file")
    parser.add_argument("--repo-name", required=True, help="Repository name (e.g., Llama-3.1-8B-Blasphemer-GGUF)")
    parser.add_argument("--username", help="HF username (optional, uses logged-in user)")
    parser.add_argument("--message", help="Commit message")
    parser.add_argument("--create-card", action="store_true", help="Create a model card")
    
    args = parser.parse_args()
    
    # Upload the file
    upload_gguf_to_hf(
        gguf_path=args.gguf_file,
        repo_name=args.repo_name,
        username=args.username,
        commit_message=args.message
    )
    
    # Create model card if requested
    if args.create_card:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        username = args.username or user_info['name']
        repo_id = f"{username}/{args.repo_name}"
        create_model_card(repo_id)
