#!/usr/bin/env bash

################################################################################
# Blasphemer GGUF Converter
# 
# Converts Blasphemer-processed models to GGUF format for LM Studio
# Developed by Christopher Bradford (@sunkencity999)
# https://github.com/sunkencity999/blasphemer
#
# Usage:
#   Interactive:  ./convert-to-gguf.sh
#   Command-line: ./convert-to-gguf.sh <model_path> [output_name] [quantization]
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'
DIM='\033[2m'

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LLAMA_CPP_DIR="$SCRIPT_DIR/llama.cpp"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
CONVERTER_SCRIPT="$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
QUANTIZER="$LLAMA_CPP_DIR/build/bin/llama-quantize"

# Default model directory
DEFAULT_MODEL_DIR="$HOME/blasphemer-models"

################################################################################
# Utility Functions
################################################################################

print_banner() {
    clear
    echo -e "${CYAN}${BOLD}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║           Blasphemer → GGUF Converter                          ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "${DIM}Convert decensored models to GGUF format for LM Studio${NC}"
    echo ""
}

print_header() {
    echo ""
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_option() {
    echo -e "  ${BOLD}[$1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Read user choice with validation
read_choice() {
    local prompt="$1"
    local max_option="$2"
    local choice
    
    while true; do
        echo ""
        read -p "$(echo -e ${BOLD}$prompt${NC}) " choice
        
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$max_option" ]; then
            echo "$choice"
            return 0
        else
            print_error "Invalid choice. Please enter a number between 1 and $max_option"
        fi
    done
}

# Read text input
read_text() {
    local prompt="$1"
    local default="$2"
    local text
    
    if [[ -n "$default" ]]; then
        read -p "$(echo -e ${BOLD}$prompt${NC}) [$default]: " text
        text=${text:-$default}
    else
        read -p "$(echo -e ${BOLD}$prompt${NC}): " text
    fi
    
    echo "$text"
}

# Read yes/no input
read_yes_no() {
    local prompt="$1"
    local default="${2:-n}"
    local answer
    
    while true; do
        if [[ "$default" == "y" ]]; then
            read -p "$(echo -e ${BOLD}$prompt${NC}) [Y/n]: " answer
            answer=${answer:-y}
        else
            read -p "$(echo -e ${BOLD}$prompt${NC}) [y/N]: " answer
            answer=${answer:-n}
        fi
        
        case "${answer,,}" in
            y|yes) echo "y"; return 0 ;;
            n|no) echo "n"; return 0 ;;
            *) print_error "Please answer 'y' or 'n'" ;;
        esac
    done
}

################################################################################
# Model Selection
################################################################################

find_models() {
    local search_dir="$1"
    local -a models=()
    
    if [[ -d "$search_dir" ]]; then
        while IFS= read -r -d '' model_dir; do
            # Check if it looks like a model (has config.json or similar)
            if [[ -f "$model_dir/config.json" ]] || [[ -f "$model_dir/model.safetensors" ]]; then
                models+=("$model_dir")
            fi
        done < <(find "$search_dir" -maxdepth 2 -type d -print0 2>/dev/null)
    fi
    
    printf '%s\n' "${models[@]}"
}

select_model_interactive() {
    print_header "Model Selection"
    
    echo "Choose how to specify the model:"
    echo ""
    print_option "1" "Browse models in $DEFAULT_MODEL_DIR"
    print_option "2" "Enter custom path"
    echo ""
    
    local choice=$(read_choice "Enter your choice (1-2):" 2)
    
    case $choice in
        1)
            # Find models in default directory
            if [[ ! -d "$DEFAULT_MODEL_DIR" ]]; then
                print_warning "Default model directory not found: $DEFAULT_MODEL_DIR"
                print_info "Creating directory..."
                mkdir -p "$DEFAULT_MODEL_DIR"
            fi
            
            echo ""
            print_info "Searching for models..."
            
            local -a models=()
            while IFS= read -r model; do
                models+=("$model")
            done < <(find_models "$DEFAULT_MODEL_DIR")
            
            if [[ ${#models[@]} -eq 0 ]]; then
                print_warning "No models found in $DEFAULT_MODEL_DIR"
                echo ""
                local custom_path=$(read_text "Enter model path manually" "")
                echo "${custom_path/#\~/$HOME}"
                return 0
            fi
            
            echo ""
            print_success "Found ${#models[@]} model(s)"
            echo ""
            
            # Display models
            for i in "${!models[@]}"; do
                local model_name=$(basename "${models[$i]}")
                local model_size=$(du -sh "${models[$i]}" 2>/dev/null | cut -f1 || echo "unknown")
                print_option "$((i+1))" "$model_name ${DIM}($model_size)${NC}"
            done
            
            print_option "$((${#models[@]}+1))" "Enter custom path"
            echo ""
            
            local model_choice=$(read_choice "Select model (1-$((${#models[@]}+1))):" "$((${#models[@]}+1))")
            
            if [[ $model_choice -le ${#models[@]} ]]; then
                echo "${models[$((model_choice-1))]}"
            else
                local custom_path=$(read_text "Enter model path" "")
                echo "${custom_path/#\~/$HOME}"
            fi
            ;;
        2)
            local custom_path=$(read_text "Enter model path" "")
            echo "${custom_path/#\~/$HOME}"
            ;;
    esac
}

select_quantization_interactive() {
    print_header "Quantization Level"
    
    echo "Select GGUF quantization level:"
    echo ""
    print_option "1" "Q4_K_M ${DIM}(~4.5GB for 7B - Recommended balance)${NC}"
    print_option "2" "Q5_K_M ${DIM}(~5.3GB for 7B - Better quality)${NC}"
    print_option "3" "Q8_0 ${DIM}(~8GB for 7B - High quality)${NC}"
    print_option "4" "F16 ${DIM}(~14GB for 7B - Full precision)${NC}"
    echo ""
    print_info "Lower quantization = smaller file, faster inference, slight quality loss"
    echo ""
    
    local choice=$(read_choice "Enter your choice (1-4):" 4)
    
    case $choice in
        1) echo "Q4_K_M" ;;
        2) echo "Q5_K_M" ;;
        3) echo "Q8_0" ;;
        4) echo "F16" ;;
    esac
}

################################################################################
# Interactive Mode
################################################################################

run_interactive() {
    print_banner
    
    # Model selection
    local model_path=$(select_model_interactive)
    
    if [[ ! -d "$model_path" ]]; then
        print_error "Model directory not found: $model_path"
        exit 1
    fi
    
    print_success "Selected: $model_path"
    
    # Output name
    local default_name=$(basename "$model_path")
    echo ""
    local output_name=$(read_text "Output filename (without extension)" "$default_name")
    
    # Quantization
    local quant_type=$(select_quantization_interactive)
    
    # Output directory
    local output_dir=$(dirname "$model_path")
    
    # Summary
    print_header "Conversion Summary"
    echo -e "${BOLD}Model:${NC} $model_path"
    echo -e "${BOLD}Output name:${NC} $output_name"
    echo -e "${BOLD}Quantization:${NC} $quant_type"
    echo -e "${BOLD}Output dir:${NC} $output_dir"
    echo ""
    
    local confirm=$(read_yes_no "Start conversion?" "y")
    
    if [[ "$confirm" != "y" ]]; then
        print_info "Conversion cancelled"
        exit 0
    fi
    
    # Run conversion
    convert_model "$model_path" "$output_name" "$quant_type"
}

################################################################################
# Conversion Logic
################################################################################

convert_model() {
    local MODEL_PATH="$1"
    local OUTPUT_NAME="$2"
    local QUANT_TYPE="$3"
    local OUTPUT_DIR=$(dirname "$MODEL_PATH")
    
    print_header "Converting Model"

    
    print_info "Model: $MODEL_PATH"
    print_info "Output: $OUTPUT_NAME"
    print_info "Quantization: $QUANT_TYPE"
    echo ""


    # Validate model path
    if [ ! -d "$MODEL_PATH" ]; then
        print_error "Model directory not found: $MODEL_PATH"
        exit 1
    fi

    # Check if required files exist
    if [ ! -f "$CONVERTER_SCRIPT" ]; then
        print_error "Converter script not found: $CONVERTER_SCRIPT"
        print_info "Make sure llama.cpp is built: cd llama.cpp && cmake -B build && cmake --build build"
        exit 1
    fi
    
    if [ ! -f "$QUANTIZER" ]; then
        print_error "Quantizer not found: $QUANTIZER"
        print_info "Build llama.cpp: cd llama.cpp && cmake -B build && cmake --build build --target llama-quantize"
        exit 1
    fi

    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    local F16_OUTPUT="$OUTPUT_DIR/${OUTPUT_NAME}-f16.gguf"
    local QUANT_OUTPUT="$OUTPUT_DIR/${OUTPUT_NAME}-${QUANT_TYPE}.gguf"

    
    echo ""
    
    # Step 1: Convert to F16 GGUF
    print_info "Step 1/2: Converting to F16 GGUF format..."
    echo ""
    
    if "$VENV_PYTHON" "$CONVERTER_SCRIPT" "$MODEL_PATH" \
        --outfile "$F16_OUTPUT" \
        --outtype f16; then
        
        echo ""
        print_success "F16 conversion complete: $F16_OUTPUT"
        
        # Get file size
        local F16_SIZE=$(du -h "$F16_OUTPUT" | cut -f1)
        print_info "Size: $F16_SIZE"
    else
        echo ""
        print_error "F16 conversion failed"
        print_info "Check the error messages above for details"
        exit 1
    fi
    
    echo ""

    # Step 2: Quantize (unless F16 was requested)
    if [ "$QUANT_TYPE" = "F16" ] || [ "$QUANT_TYPE" = "f16" ]; then
        print_header "Conversion Complete!"
        print_success "F16 model ready: $F16_OUTPUT"
        echo ""
        echo -e "${BOLD}To use in LM Studio:${NC}"
        echo "  1. Open LM Studio"
        echo "  2. The model should appear automatically in 'My Models'"
        echo "  3. Or click 'Load Model' and select: $F16_OUTPUT"
        echo ""
    else
        print_info "Step 2/2: Quantizing to $QUANT_TYPE..."
        echo ""
        
        if "$QUANTIZER" "$F16_OUTPUT" "$QUANT_OUTPUT" "$QUANT_TYPE"; then
            echo ""
            print_success "Quantization complete: $QUANT_OUTPUT"
            
            # Get file sizes
            local QUANT_SIZE=$(du -h "$QUANT_OUTPUT" | cut -f1)
            print_info "Size: $QUANT_SIZE"
            
            # Optionally remove F16 file to save space
            echo ""
            local remove_f16=$(read_yes_no "Remove intermediate F16 file to save space?" "n")
            
            if [[ "$remove_f16" == "y" ]]; then
                rm "$F16_OUTPUT"
                print_success "Removed F16 file (saved $(du -h "$F16_OUTPUT" 2>/dev/null | cut -f1 || echo 'space'))"
            else
                print_info "Kept F16 file: $F16_OUTPUT"
            fi
            
            print_header "Conversion Complete!"
            
            echo -e "${BOLD}Output file:${NC}"
            echo "  $QUANT_OUTPUT ($QUANT_SIZE)"
            echo ""
            echo -e "${BOLD}To use in LM Studio:${NC}"
            echo "  1. Open LM Studio"
            echo "  2. The model should appear automatically in 'My Models'"
            echo "  3. Or click 'Load Model' and select: $QUANT_OUTPUT"
            echo ""
            echo -e "${CYAN}Tip:${NC} Try different quantization levels for quality vs. size balance"
            echo ""
        else
            echo ""
            print_error "Quantization failed"
            print_info "Check the error messages above for details"
            print_info "The F16 file is still available at: $F16_OUTPUT"
            exit 1
        fi
    fi
}

################################################################################
# Main Entry Point
################################################################################

main() {
    # Check if running in interactive mode (no arguments)
    if [ $# -eq 0 ]; then
        run_interactive
    else
        # Command-line mode for backward compatibility
        if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
            print_banner
            echo -e "${BOLD}Usage:${NC}"
            echo "  Interactive:  $0"
            echo "  Command-line: $0 <model_path> [output_name] [quantization]"
            echo ""
            echo -e "${BOLD}Arguments:${NC}"
            echo "  model_path     : Path to Blasphemer model directory (required)"
            echo "  output_name    : Output filename without extension (optional)"
            echo "  quantization   : Quantization type (default: Q4_K_M)"
            echo ""
            echo -e "${BOLD}Quantization types:${NC}"
            echo "  Q4_K_M  : Balanced quality/size (4-bit, recommended)"
            echo "  Q5_K_M  : Better quality (5-bit)"
            echo "  Q8_0    : High quality (8-bit)"
            echo "  F16     : Full precision (16-bit)"
            echo ""
            echo -e "${BOLD}Examples:${NC}"
            echo "  $0"
            echo "  $0 ~/blasphemer-models/Llama-3.1-8B-Instruct-blasphemer"
            echo "  $0 ~/blasphemer-models/Llama-3.1-8B-Instruct-blasphemer my-model Q5_K_M"
            echo ""
            exit 0
        fi
        
        local MODEL_PATH="$1"
        local OUTPUT_NAME="${2:-$(basename "$MODEL_PATH")}"
        local QUANT_TYPE="${3:-Q4_K_M}"
        
        # Expand tilde
        MODEL_PATH="${MODEL_PATH/#\~/$HOME}"
        
        print_banner
        convert_model "$MODEL_PATH" "$OUTPUT_NAME" "$QUANT_TYPE"
    fi
}

# Run main
main "$@"
