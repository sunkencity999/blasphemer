# Documentation Changelog

## 2025-11-18: Major Consolidation

### Changes

Consolidated 8 separate documentation files into one comprehensive **USER_GUIDE.md**.

### Files Removed

The following files were removed from the root directory:

1. `INSTALLATION.md` - Installation and setup instructions
2. `QUICK_START.md` - Quick start workflow
3. `RECOMMENDED_MODELS.md` - Model recommendations
4. `LLAMA_CPP_SETUP.md` - llama.cpp details
5. `CHECKPOINT_GUIDE.md` - Checkpoint system documentation
6. `IMPROVEMENTS_SUMMARY.md` - Technical improvements summary
7. `PROPOSED_IMPROVEMENTS.md` - Future enhancement ideas
8. `README_IMPROVEMENTS.md` - Quick reference for improvements

### Current Documentation Structure

**Primary Documentation**:
- `USER_GUIDE.md` - Complete user guide (all topics consolidated)
- `README.md` - Original Heretic project README (preserved)

**Configuration**:
- `config.default.toml` - Default configuration with documentation

**Scripts**:
- `convert-to-gguf.sh` - GGUF conversion helper (includes inline help)

### USER_GUIDE.md Sections

1. Introduction
2. Installation
3. Quick Start
4. Model Recommendations
5. LM Studio Integration (GGUF conversion)
6. Checkpoint & Resume System
7. Configuration
8. Troubleshooting
9. Advanced Usage
10. Resources

### Benefits

- Single source of truth for all documentation
- Easier to maintain and update
- Professional presentation
- Clean root directory
- Comprehensive table of contents
- Searchable content

### Usage

For all usage questions, refer to:

```bash
cat USER_GUIDE.md
```

Or view specific sections using the table of contents.
