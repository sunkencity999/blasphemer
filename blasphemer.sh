#!/usr/bin/env bash

################################################################################
# Blasphemer Launcher Script
# 
# Interactive menu-driven interface for Blasphemer operations
# Developed by Christopher Bradford (@sunkencity999)
# https://github.com/sunkencity999/blasphemer
################################################################################

set -euo pipefail

# Error handler
trap 'echo "Error on line $LINENO. Exit code: $?" >&2' ERR

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

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default paths
DEFAULT_MODEL_DIR="$HOME/blasphemer-models"
VENV_DIR="$SCRIPT_DIR/venv"

################################################################################
# Utility Functions
################################################################################

print_banner() {
    clear
    printf "%b%b\n" "${CYAN}" "${BOLD}"
    printf "█▀▄░█░░░█▀█░█▀▀░█▀█░█░█░█▀▀░█▄█░█▀▀░█▀▄\n"
    printf "█▀▄░█░░░█▀█░▀▀█░█▀▀░█▀█░█▀▀░█░█░█▀▀░█▀▄\n"
    printf "▀▀░░▀▀▀░▀░▀░▀▀▀░▀░░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░▀\n"
    printf "%b\n" "${NC}"
    printf "%bDeveloped by Christopher Bradford (@sunkencity999)%b\n" "${DIM}" "${NC}"
    printf "%bEnhanced fork of Heretic - optimized for macOS%b\n" "${DIM}" "${NC}"
    printf "\n"
}

print_header() {
    printf "\n"
    printf "%b%b═══════════════════════════════════════════════════════════════%b\n" "${CYAN}" "${BOLD}" "${NC}"
    printf "%b%b  %s%b\n" "${CYAN}" "${BOLD}" "$1" "${NC}"
    printf "%b%b═══════════════════════════════════════════════════════════════%b\n" "${CYAN}" "${BOLD}" "${NC}"
    printf "\n"
}

print_option() {
    printf "  %b[%s]%b %b\n" "${BOLD}" "$1" "${NC}" "$2"
}

print_success() {
    printf "%b✓%b %s\n" "${GREEN}" "${NC}" "$1"
}

print_error() {
    printf "%b✗%b %s\n" "${RED}" "${NC}" "$1"
}

print_info() {
    printf "%bℹ%b %s\n" "${CYAN}" "${NC}" "$1"
}

print_warning() {
    printf "%b⚠%b %s\n" "${YELLOW}" "${NC}" "$1"
}

# Read user input with validation
read_choice() {
    local prompt="$1"
    local max_option="$2"
    local choice
    
    while true; do
        printf "\n"
        read -p "$(printf "%b%s%b" "${BOLD}" "$prompt" "${NC}") " choice
        
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$max_option" ]; then
            printf "%s" "$choice"
            return 0
        else
            print_error "Invalid choice. Please enter a number between 1 and $max_option"
        fi
    done
}

# Read yes/no input
read_yes_no() {
    local prompt="$1"
    local default="${2:-n}"
    local answer
    
    while true; do
        if [[ "$default" == "y" ]]; then
            read -p "$(printf "%b%s%b" "${BOLD}" "$prompt" "${NC}") [Y/n]: " answer
            answer=${answer:-y}
        else
            read -p "$(printf "%b%s%b" "${BOLD}" "$prompt" "${NC}") [y/N]: " answer
            answer=${answer:-n}
        fi
        
        case "${answer,,}" in
            y|yes) printf "y"; return 0 ;;
            n|no) printf "n"; return 0 ;;
            *) print_error "Please answer 'y' or 'n'" ;;
        esac
    done
}

# Read text input
read_text() {
    local prompt="$1"
    local default="$2"
    local text
    
    if [[ -n "$default" ]]; then
        read -p "$(printf "%b%s%b" "${BOLD}" "$prompt" "${NC}") [$default]: " text
        text=${text:-$default}
    else
        read -p "$(printf "%b%s%b" "${BOLD}" "$prompt" "${NC}"): " text
    fi
    
    printf "%s" "$text"
}

################################################################################
# Environment Setup
################################################################################

setup_environment() {
    print_header "Setting Up Environment"
    
    # Check if virtual environment exists
    if [[ ! -d "$VENV_DIR" ]]; then
        print_warning "Virtual environment not found"
        print_info "Creating virtual environment..."
        
        if python3 -m venv "$VENV_DIR"; then
            print_success "Virtual environment created"
        else
            print_error "Failed to create virtual environment"
            print_error "Please run: python3 -m venv venv"
            exit 1
        fi
    fi
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        # Use set +euo pipefail temporarily for source
        set +u
        source "$VENV_DIR/bin/activate" || {
            print_error "Failed to activate virtual environment"
            exit 1
        }
        set -u
        print_success "Environment activated"
    else
        print_error "Virtual environment activation script not found"
        exit 1
    fi
    
    # Check if blasphemer is installed
    if ! command -v blasphemer &> /dev/null; then
        print_warning "Blasphemer not installed in virtual environment"
        print_info "Installing Blasphemer..."
        
        if pip install -e . > /dev/null 2>&1; then
            print_success "Blasphemer installed"
        else
            print_error "Failed to install Blasphemer"
            print_error "Please run: pip install -e ."
            exit 1
        fi
    fi
    
    # Check MPS availability (non-fatal)
    print_info "Checking GPU availability..."
    if python3 -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
        print_success "Apple Silicon MPS GPU detected"
    else
        print_warning "MPS not available - will use CPU (slower)"
    fi
    
    printf "\n"
}

################################################################################
# Model Selection
################################################################################

show_recommended_models() {
    print_header "Recommended Models"
    
    printf "%bQuick Testing (15-20 minutes):%b\n" "${BOLD}" "${NC}"
    printf "  • microsoft/Phi-3-mini-4k-instruct (3.8B)\n"
    printf "  • microsoft/Phi-3-mini-128k-instruct (3.8B)\n"
    printf "\n"
    printf "%bGood Quality (30-60 minutes):%b\n" "${BOLD}" "${NC}"
    printf "  • Qwen/Qwen2.5-7B-Instruct\n"
    printf "  • meta-llama/Llama-3.1-8B-Instruct\n"
    printf "  • mistralai/Mistral-7B-Instruct-v0.3\n"
    printf "\n"
    printf "%bHigh Quality (60-90+ minutes):%b\n" "${BOLD}" "${NC}"
    printf "  • Qwen/Qwen2.5-14B-Instruct\n"
    printf "  • meta-llama/Llama-3.1-70B-Instruct (requires 64GB+ RAM)\n"
    printf "\n"
    print_info "See USER_GUIDE.md for complete model recommendations"
}

select_model() {
    print_header "Model Selection" >&2
    
    echo "Choose an option:" >&2
    echo "" >&2
    print_option "1" "Use a recommended model" >&2
    print_option "2" "Enter a custom model name" >&2
    print_option "3" "Show recommended models first" >&2
    echo "" >&2
    
    local choice=$(read_choice "Enter your choice (1-3):" 3)
    
    case $choice in
        1)
            echo "" >&2
            echo "Select a recommended model:" >&2
            echo "" >&2
            print_option "1" "microsoft/Phi-3-mini-4k-instruct ${DIM}(3.8B - Fast, good for testing)${NC}" >&2
            print_option "2" "Qwen/Qwen2.5-7B-Instruct ${DIM}(7B - Excellent quality)${NC}" >&2
            print_option "3" "meta-llama/Llama-3.1-8B-Instruct ${DIM}(8B - Most popular)${NC}" >&2
            print_option "4" "mistralai/Mistral-7B-Instruct-v0.3 ${DIM}(7B - High quality)${NC}" >&2
            print_option "5" "Qwen/Qwen2.5-14B-Instruct ${DIM}(14B - Best quality)${NC}" >&2
            print_option "6" "Enter custom model" >&2
            echo "" >&2
            
            local model_choice=$(read_choice "Enter your choice (1-6):" 6)
            
            case $model_choice in
                1) printf "microsoft/Phi-3-mini-4k-instruct" ;;
                2) printf "Qwen/Qwen2.5-7B-Instruct" ;;
                3) printf "meta-llama/Llama-3.1-8B-Instruct" ;;
                4) printf "mistralai/Mistral-7B-Instruct-v0.3" ;;
                5) printf "Qwen/Qwen2.5-14B-Instruct" ;;
                6) read_text "Enter model name (e.g., meta-llama/Llama-3.1-8B-Instruct)" "" ;;
            esac
            ;;
        2)
            read_text "Enter model name (e.g., meta-llama/Llama-3.1-8B-Instruct)" ""
            ;;
        3)
            show_recommended_models
            echo ""
            read -p "Press Enter to continue..."
            select_model
            ;;
    esac
}

################################################################################
# Save Location
################################################################################

select_save_location() {
    local model_name="$1"
    local model_basename=$(basename "$model_name")
    
    print_header "Save Location" >&2
    
    # Ensure default directory exists
    mkdir -p "$DEFAULT_MODEL_DIR"
    
    echo "Where should the decensored model be saved?" >&2
    echo "" >&2
    print_option "1" "Default location: ${DIM}$DEFAULT_MODEL_DIR/$model_basename-blasphemer${NC}" >&2
    print_option "2" "Custom location" >&2
    echo "" >&2
    
    local choice=$(read_choice "Enter your choice (1-2):" 2)
    
    case $choice in
        1)
            printf "%s" "$DEFAULT_MODEL_DIR/$model_basename-blasphemer"
            ;;
        2)
            local custom_path=$(read_text "Enter full path for model" "$HOME/")
            # Expand tilde
            custom_path="${custom_path/#\~/$HOME}"
            printf "%s" "$custom_path"
            ;;
    esac
}

################################################################################
# Quantization Selection
################################################################################

select_quantization() {
    print_header "Quantization Level" >&2
    
    echo "Select GGUF quantization level:" >&2
    echo "" >&2
    print_option "1" "Q4_K_M ${DIM}(~4.5GB for 7B model - Recommended balance)${NC}" >&2
    print_option "2" "Q5_K_M ${DIM}(~5.3GB for 7B model - Better quality)${NC}" >&2
    print_option "3" "Q8_0 ${DIM}(~8GB for 7B model - High quality)${NC}" >&2
    print_option "4" "F16 ${DIM}(~14GB for 7B model - Full precision)${NC}" >&2
    print_option "5" "Skip GGUF conversion" >&2
    echo "" >&2
    print_info "Lower quantization = smaller file size, faster inference, slight quality loss" >&2
    echo "" >&2
    
    local choice=$(read_choice "Enter your choice (1-5):" 5)
    
    case $choice in
        1) printf "Q4_K_M" ;;
        2) printf "Q5_K_M" ;;
        3) printf "Q8_0" ;;
        4) printf "F16" ;;
        5) printf "SKIP" ;;
    esac
}

################################################################################
# Advanced Options
################################################################################

configure_advanced_options() {
    print_header "Advanced Options" >&2
    
    echo "Configure advanced settings:" >&2
    echo "" >&2
    
    # Number of trials
    local n_trials
    local use_default=$(read_yes_no "Use default settings? (200 trials, auto batch size)" "y")
    
    if [[ "$use_default" == "y" ]]; then
        printf "200\n"  # default trials
        printf "0\n"    # auto batch size
        printf "n"    # no resume
        return 0
    fi
    
    echo "" >&2
    n_trials=$(read_text "Number of optimization trials" "200")
    
    echo "" >&2
    local batch_size=$(read_text "Batch size (0 for auto-detect)" "0")
    
    echo "" >&2
    local resume=$(read_yes_no "Resume from existing checkpoint if available?" "n")
    
    printf "%s\n" "$n_trials"
    printf "%s\n" "$batch_size"
    printf "%s" "$resume"
}

################################################################################
# Operation Selection
################################################################################

select_operation() {
    print_header "Operation" >&2
    
    echo "What would you like to do?" >&2
    echo "" >&2
    print_option "1" "Process a new model ${DIM}(Full workflow: decensor + save + convert)${NC}" >&2
    print_option "2" "Process model only ${DIM}(Decensor without conversion)${NC}" >&2
    print_option "3" "Convert existing model to GGUF ${DIM}(Already decensored)${NC}" >&2
    print_option "4" "Resume interrupted processing ${DIM}(Continue from checkpoint)${NC}" >&2
    print_option "5" "View help and documentation" >&2
    print_option "6" "Exit" >&2
    echo "" >&2
    
    local choice=$(read_choice "Enter your choice (1-6):" 6)
    printf "%s" "$choice"
}

################################################################################
# Processing Functions
################################################################################

process_new_model() {
    print_banner
    
    print_info "DEBUG: Starting process_new_model()" >&2
    
    # Model selection
    print_info "DEBUG: Calling select_model()" >&2
    local model_name=$(select_model)
    print_info "DEBUG: select_model returned: '$model_name'" >&2
    
    if [[ -z "$model_name" ]]; then
        print_error "No model selected"
        return 1
    fi
    
    print_success "Selected model: $model_name"
    
    # Save location
    local save_path=$(select_save_location "$model_name")
    print_success "Save location: $save_path"
    
    # Advanced options
    print_info "Configuring advanced options..."
    local options=$(configure_advanced_options)
    local n_trials=$(echo "$options" | sed -n '1p')
    local batch_size=$(echo "$options" | sed -n '2p')
    local resume=$(echo "$options" | sed -n '3p')
    
    # Quantization
    local quant_type=$(select_quantization)
    
    # Summary
    print_header "Processing Summary"
    printf "%bModel:%b %s\n" "${BOLD}" "${NC}" "$model_name"
    printf "%bSave to:%b %s\n" "${BOLD}" "${NC}" "$save_path"
    printf "%bTrials:%b %s\n" "${BOLD}" "${NC}" "$n_trials"
    printf "%bBatch size:%b %s\n" "${BOLD}" "${NC}" "$([ "$batch_size" -eq 0 ] && echo "Auto" || echo "$batch_size")"
    printf "%bQuantization:%b %s\n" "${BOLD}" "${NC}" "$quant_type"
    printf "\n"
    
    local confirm=$(read_yes_no "Start processing?" "y")
    
    if [[ "$confirm" != "y" ]]; then
        print_info "Operation cancelled"
        return 0
    fi
    
    # Build command
    local cmd="blasphemer --model '$model_name' --n-trials $n_trials"
    
    if [[ "$batch_size" != "0" ]]; then
        cmd="$cmd --max-batch-size $batch_size"
    fi
    
    if [[ "$resume" == "y" ]]; then
        cmd="$cmd --resume"
    fi
    
    print_header "Processing Model"
    print_info "Command: $cmd"
    echo ""
    print_info "This will take some time. You can press Ctrl+C to interrupt safely."
    print_info "Progress is automatically checkpointed every trial."
    echo ""
    sleep 2
    
    # Run blasphemer
    eval $cmd || {
        print_error "Processing failed or was interrupted"
        print_info "You can resume with the 'Resume interrupted processing' option"
        return 1
    }
    
    # Note: User will interactively save the model through Blasphemer's prompts
    # We need to ask them where they saved it
    echo ""
    print_success "Model processing complete!"
    echo ""
    
    local saved_path=$(read_text "Enter the path where you saved the model" "$save_path")
    
    # GGUF conversion
    if [[ "$quant_type" != "SKIP" ]] && [[ -d "$saved_path" ]]; then
        convert_to_gguf "$saved_path" "$quant_type"
    else
        print_info "Skipping GGUF conversion"
    fi
    
    print_success "All done!"
}

process_model_only() {
    print_banner
    
    local model_name=$(select_model)
    print_success "Selected model: $model_name"
    
    local options=$(configure_advanced_options)
    local n_trials=$(echo "$options" | sed -n '1p')
    local batch_size=$(echo "$options" | sed -n '2p')
    local resume=$(echo "$options" | sed -n '3p')
    
    print_header "Processing Summary"
    printf "%bModel:%b %s\n" "${BOLD}" "${NC}" "$model_name"
    printf "%bTrials:%b %s\n" "${BOLD}" "${NC}" "$n_trials"
    printf "%bBatch size:%b %s\n" "${BOLD}" "${NC}" "$([ "$batch_size" -eq 0 ] && echo "Auto" || echo "$batch_size")"
    printf "\n"
    
    local confirm=$(read_yes_no "Start processing?" "y")
    
    if [[ "$confirm" != "y" ]]; then
        print_info "Operation cancelled"
        return 0
    fi
    
    local cmd="blasphemer --model '$model_name' --n-trials $n_trials"
    
    if [[ "$batch_size" != "0" ]]; then
        cmd="$cmd --max-batch-size $batch_size"
    fi
    
    if [[ "$resume" == "y" ]]; then
        cmd="$cmd --resume"
    fi
    
    print_header "Processing Model"
    print_info "Command: $cmd"
    echo ""
    
    eval $cmd
}

convert_to_gguf() {
    local model_path="$1"
    local quant_type="${2:-Q4_K_M}"
    
    print_header "Converting to GGUF"
    
    if [[ ! -d "$model_path" ]]; then
        print_error "Model directory not found: $model_path"
        return 1
    fi
    
    print_info "Model: $model_path"
    print_info "Quantization: $quant_type"
    echo ""
    
    if [[ ! -x "./convert-to-gguf.sh" ]]; then
        print_error "Conversion script not found or not executable"
        return 1
    fi
    
    print_info "Starting conversion..."
    echo ""
    
    ./convert-to-gguf.sh "$model_path" "" "$quant_type" || {
        print_error "Conversion failed"
        return 1
    }
    
    print_success "Conversion complete!"
}

convert_existing_model() {
    print_banner
    print_header "Convert Existing Model to GGUF"
    
    echo ""
    local model_path=$(read_text "Enter path to model directory" "$DEFAULT_MODEL_DIR/")
    model_path="${model_path/#\~/$HOME}"
    
    if [[ ! -d "$model_path" ]]; then
        print_error "Directory not found: $model_path"
        return 1
    fi
    
    local quant_type=$(select_quantization)
    
    if [[ "$quant_type" == "SKIP" ]]; then
        print_info "Operation cancelled"
        return 0
    fi
    
    convert_to_gguf "$model_path" "$quant_type"
}

resume_processing() {
    print_banner
    print_header "Resume Interrupted Processing"
    
    local model_name=$(select_model)
    
    print_info "Looking for checkpoint for: $model_name"
    echo ""
    
    local confirm=$(read_yes_no "Resume processing?" "y")
    
    if [[ "$confirm" != "y" ]]; then
        print_info "Operation cancelled"
        return 0
    fi
    
    print_info "Resuming from checkpoint..."
    echo ""
    
    blasphemer --resume --model "$model_name"
}

show_help() {
    print_banner
    print_header "Help & Documentation"
    
    echo "Available documentation:"
    echo ""
    echo "  ${BOLD}USER_GUIDE.md${NC} - Complete user guide"
    echo "  ${BOLD}README.md${NC} - Project overview and quick start"
    echo "  ${BOLD}config.default.toml${NC} - Configuration reference"
    echo ""
    echo "Quick commands:"
    echo ""
    echo "  ${BOLD}blasphemer --help${NC} - View all command-line options"
    echo "  ${BOLD}./blasphemer.sh${NC} - This interactive launcher"
    echo "  ${BOLD}./convert-to-gguf.sh${NC} - GGUF conversion helper"
    echo ""
    echo "Common tasks:"
    echo ""
    echo "  ${DIM}Process a model:${NC}"
    echo "    blasphemer microsoft/Phi-3-mini-4k-instruct"
    echo ""
    echo "  ${DIM}Resume interrupted run:${NC}"
    echo "    blasphemer --resume microsoft/Phi-3-mini-4k-instruct"
    echo ""
    echo "  ${DIM}Convert to GGUF:${NC}"
    echo "    ./convert-to-gguf.sh ~/models/model-name"
    echo ""
    
    read -p "Press Enter to continue..."
}

################################################################################
# Main Menu Loop
################################################################################

main() {
    # Setup environment
    printf "DEBUG: Starting main()\\n" >&2
    setup_environment
    printf "DEBUG: Environment setup complete\\n" >&2
    
    # Main loop
    while true; do
        print_banner
        
        local operation=$(select_operation)
        printf "DEBUG: Selected operation: '%s'\\n" "$operation" >&2
        
        case $operation in
            1) 
                printf "DEBUG: Calling process_new_model()\\n" >&2
                process_new_model
                printf "DEBUG: process_new_model() returned: %s\\n" "$?" >&2
                ;;
            2) process_model_only ;;
            3) convert_existing_model ;;
            4) resume_processing ;;
            5) show_help ;;
            6)
                echo ""
                print_success "Thank you for using Blasphemer!"
                echo ""
                exit 0
                ;;
            *)
                printf "DEBUG: Unknown operation: '%s'\\n" "$operation" >&2
                ;;
        esac
        
        printf "DEBUG: Reached end of case statement\\n" >&2
        echo ""
        read -p "Press Enter to return to main menu..."
    done
}

# Run main
main "$@"
