#!/bin/bash
# Heretic Model to GGUF Converter
# Converts Heretic-processed models to GGUF format for LM Studio

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LLAMA_CPP_DIR="$SCRIPT_DIR/llama.cpp"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
CONVERTER_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
QUANTIZER="$LLAMA_CPP_DIR/build/bin/llama-quantize"

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Heretic Model → GGUF Converter           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo

# Check arguments
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 <model_path> [output_name] [quantization]"
    echo
    echo -e "${YELLOW}Arguments:${NC}"
    echo "  model_path     : Path to Heretic model directory (required)"
    echo "  output_name    : Output filename without extension (optional)"
    echo "  quantization   : Quantization type (default: Q4_K_M)"
    echo
    echo -e "${YELLOW}Common quantization types:${NC}"
    echo "  Q4_K_M  : Good balance (4-bit, recommended)"
    echo "  Q5_K_M  : Better quality (5-bit)"
    echo "  Q8_0    : High quality (8-bit)"
    echo "  F16     : Full precision (16-bit)"
    echo
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 ~/heretic-models/llama-heretic"
    echo "  $0 ~/heretic-models/llama-heretic my-model Q5_K_M"
    exit 1
fi

MODEL_PATH="$1"
OUTPUT_DIR="$(dirname "$MODEL_PATH")"

# Expand tilde to home directory
MODEL_PATH="${MODEL_PATH/#\~/$HOME}"
OUTPUT_DIR="${OUTPUT_DIR/#\~/$HOME}"

# Determine output name
if [ -n "$2" ]; then
    OUTPUT_NAME="$2"
else
    OUTPUT_NAME="$(basename "$MODEL_PATH")"
fi

# Determine quantization type
QUANT_TYPE="${3:-Q4_K_M}"

# Validate model path
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model directory not found: $MODEL_PATH${NC}"
    exit 1
fi

# Check if required files exist
if [ ! -f "$CONVERTER_SCRIPT" ]; then
    echo -e "${RED}Error: Converter script not found at $CONVERTER_SCRIPT${NC}"
    exit 1
fi

if [ ! -f "$QUANTIZER" ]; then
    echo -e "${RED}Error: Quantizer not found at $QUANTIZER${NC}"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

F16_OUTPUT="$OUTPUT_DIR/${OUTPUT_NAME}-f16.gguf"
QUANT_OUTPUT="$OUTPUT_DIR/${OUTPUT_NAME}-${QUANT_TYPE}.gguf"

echo -e "${GREEN}Configuration:${NC}"
echo "  Model path:    $MODEL_PATH"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Output name:   $OUTPUT_NAME"
echo "  Quantization:  $QUANT_TYPE"
echo

# Step 1: Convert to F16 GGUF
echo -e "${BLUE}[1/2] Converting to F16 GGUF format...${NC}"
"$VENV_PYTHON" "$CONVERTER_SCRIPT" "$MODEL_PATH" \
    --outfile "$F16_OUTPUT" \
    --outtype f16

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Conversion successful: $F16_OUTPUT${NC}"
    
    # Get file size
    F16_SIZE=$(du -h "$F16_OUTPUT" | cut -f1)
    echo -e "  Size: ${GREEN}$F16_SIZE${NC}"
else
    echo -e "${RED}✗ Conversion failed${NC}"
    exit 1
fi

echo

# Step 2: Quantize (unless F16 was requested)
if [ "$QUANT_TYPE" = "F16" ] || [ "$QUANT_TYPE" = "f16" ]; then
    echo -e "${GREEN}✓ F16 model ready: $F16_OUTPUT${NC}"
    echo
    echo -e "${BLUE}To use in LM Studio:${NC}"
    echo "  1. Open LM Studio"
    echo "  2. The model should appear automatically in your models list"
    echo "  3. Or use 'Load Model' → select: $F16_OUTPUT"
else
    echo -e "${BLUE}[2/2] Quantizing to $QUANT_TYPE...${NC}"
    "$QUANTIZER" "$F16_OUTPUT" "$QUANT_OUTPUT" "$QUANT_TYPE"
    
    if [ $? -eq 0 ]; then
        echo
        echo -e "${GREEN}✓ Quantization successful: $QUANT_OUTPUT${NC}"
        
        # Get file sizes
        QUANT_SIZE=$(du -h "$QUANT_OUTPUT" | cut -f1)
        echo -e "  Size: ${GREEN}$QUANT_SIZE${NC}"
        
        # Optionally remove F16 file to save space
        echo
        read -p "Remove intermediate F16 file to save space? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm "$F16_OUTPUT"
            echo -e "${GREEN}✓ Removed F16 file${NC}"
        else
            echo -e "${YELLOW}Kept F16 file: $F16_OUTPUT${NC}"
        fi
    else
        echo -e "${RED}✗ Quantization failed${NC}"
        exit 1
    fi
    
    echo
    echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  Conversion Complete!                      ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
    echo
    echo -e "${BLUE}Output file:${NC}"
    echo "  $QUANT_OUTPUT"
    echo
    echo -e "${BLUE}To use in LM Studio:${NC}"
    echo "  1. Open LM Studio"
    echo "  2. The model should appear automatically in your models list"
    echo "  3. Or use 'Load Model' → select: $QUANT_OUTPUT"
fi

echo
echo -e "${YELLOW}Tip:${NC} For different quality levels, try:"
echo "  - Q4_K_M: Balanced (smaller, faster)"
echo "  - Q5_K_M: Better quality (larger, slower)"
echo "  - Q8_0:   High quality (much larger)"
