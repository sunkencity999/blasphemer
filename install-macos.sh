#!/usr/bin/env bash

################################################################################
# Blasphemer Installation Script for macOS
# 
# Developed by Christopher Bradford (@sunkencity999)
# https://github.com/sunkencity999/blasphemer
#
# This script:
# - Checks all prerequisites
# - Installs Blasphemer and dependencies
# - Builds llama.cpp for GGUF conversion
# - Tracks progress and can resume if interrupted
# - Provides verbose output with clear error messages
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Installation state file
STATE_FILE=".blasphemer_install_state"
INSTALL_DIR="${INSTALL_DIR:-$HOME/blasphemer}"
REPO_URL="https://github.com/sunkencity999/blasphemer.git"

################################################################################
# Utility Functions
################################################################################

print_header() {
    echo ""
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}▶${NC} ${BOLD}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "  ${CYAN}ℹ${NC} $1"
}

# Save installation state
save_state() {
    local step=$1
    echo "$step" > "$STATE_FILE"
    print_info "Progress saved: $step"
}

# Load installation state
load_state() {
    if [[ -f "$STATE_FILE" ]]; then
        cat "$STATE_FILE"
    else
        echo "start"
    fi
}

# Check if a step is already completed
is_step_completed() {
    local step=$1
    local current_state=$(load_state)
    
    case "$current_state" in
        "start") return 1 ;;
        "prerequisites_checked") [[ "$step" == "prerequisites_checked" ]] && return 0 || return 1 ;;
        "repo_cloned") [[ "$step" =~ ^(prerequisites_checked|repo_cloned)$ ]] && return 0 || return 1 ;;
        "venv_created") [[ "$step" =~ ^(prerequisites_checked|repo_cloned|venv_created)$ ]] && return 0 || return 1 ;;
        "dependencies_installed") [[ "$step" =~ ^(prerequisites_checked|repo_cloned|venv_created|dependencies_installed)$ ]] && return 0 || return 1 ;;
        "llama_cpp_built") [[ "$step" =~ ^(prerequisites_checked|repo_cloned|venv_created|dependencies_installed|llama_cpp_built)$ ]] && return 0 || return 1 ;;
        "completed") return 0 ;;
        *) return 1 ;;
    esac
}

################################################################################
# Prerequisite Checks
################################################################################

check_prerequisites() {
    if is_step_completed "prerequisites_checked"; then
        print_success "Prerequisites already checked (resuming installation)"
        return 0
    fi
    
    print_header "Checking Prerequisites"
    
    local all_ok=true
    
    # Check macOS
    print_step "Checking operating system..."
    if [[ "$(uname)" != "Darwin" ]]; then
        print_error "This script is designed for macOS only"
        all_ok=false
    else
        print_success "Running on macOS $(sw_vers -productVersion)"
    fi
    
    # Check architecture
    print_step "Checking architecture..."
    local arch=$(uname -m)
    if [[ "$arch" == "arm64" ]]; then
        print_success "Apple Silicon (ARM64) detected"
    elif [[ "$arch" == "x86_64" ]]; then
        print_warning "Intel (x86_64) detected - Apple Silicon recommended for best performance"
    else
        print_error "Unknown architecture: $arch"
        all_ok=false
    fi
    
    # Check Python version
    print_step "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version | awk '{print $2}')
        local major=$(echo "$python_version" | cut -d. -f1)
        local minor=$(echo "$python_version" | cut -d. -f2)
        
        if [[ $major -ge 3 ]] && [[ $minor -ge 10 ]]; then
            print_success "Python $python_version found (required: 3.10+)"
        else
            print_error "Python 3.10+ required, found: $python_version"
            print_info "Install with: brew install python@3.12"
            all_ok=false
        fi
    else
        print_error "Python 3 not found"
        print_info "Install with: brew install python@3.12"
        all_ok=false
    fi
    
    # Check pip
    print_step "Checking pip..."
    if python3 -m pip --version &> /dev/null; then
        print_success "pip found"
    else
        print_error "pip not found"
        print_info "Install with: python3 -m ensurepip --upgrade"
        all_ok=false
    fi
    
    # Check git
    print_step "Checking git..."
    if command -v git &> /dev/null; then
        local git_version=$(git --version | awk '{print $3}')
        print_success "git $git_version found"
    else
        print_error "git not found"
        print_info "Install with: xcode-select --install"
        all_ok=false
    fi
    
    # Check cmake
    print_step "Checking cmake..."
    if command -v cmake &> /dev/null; then
        local cmake_version=$(cmake --version | head -n1 | awk '{print $3}')
        print_success "cmake $cmake_version found"
    else
        print_error "cmake not found"
        print_info "Install with: brew install cmake"
        all_ok=false
    fi
    
    # Check for Homebrew (optional but recommended)
    print_step "Checking Homebrew (optional)..."
    if command -v brew &> /dev/null; then
        print_success "Homebrew found"
    else
        print_warning "Homebrew not found (recommended for easy dependency management)"
        print_info "Install from: https://brew.sh"
    fi
    
    # Check disk space
    print_step "Checking disk space..."
    local available_space=$(df -h "$HOME" | awk 'NR==2 {print $4}' | sed 's/Gi//')
    if [[ ${available_space%.*} -ge 10 ]]; then
        print_success "Sufficient disk space available (${available_space}GB free)"
    else
        print_warning "Low disk space: ${available_space}GB free (10GB+ recommended)"
    fi
    
    echo ""
    
    if [[ "$all_ok" == false ]]; then
        print_error "Prerequisites check failed. Please install missing dependencies and try again."
        exit 1
    fi
    
    print_success "All prerequisites satisfied!"
    save_state "prerequisites_checked"
}

################################################################################
# Repository Cloning
################################################################################

clone_repository() {
    if is_step_completed "repo_cloned"; then
        print_success "Repository already cloned (resuming installation)"
        return 0
    fi
    
    print_header "Cloning Blasphemer Repository"
    
    # Check if directory already exists
    if [[ -d "$INSTALL_DIR" ]]; then
        # Check if this script is running from inside the blasphemer directory
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        SCRIPT_IN_BLASPHEMER=false
        if [[ "$SCRIPT_DIR" == "$INSTALL_DIR"* ]]; then
            SCRIPT_IN_BLASPHEMER=true
        fi
        
        print_warning "Directory $INSTALL_DIR already exists"
        
        # Check if it's a git repository
        if [[ -d "$INSTALL_DIR/.git" ]]; then
            print_info "Existing installation detected"
            echo ""
            echo "Options:"
            echo "  1) Update existing installation (git pull)"
            echo "  2) Keep existing files and continue installation"
            echo "  3) Remove and re-clone (fresh install)"
            echo "  4) Cancel"
            echo ""
            read -p "Choose option [1-4]: " -n 1 -r
            echo
            
            case $REPLY in
                1)
                    print_step "Updating repository..."
                    cd "$INSTALL_DIR"
                    if git pull && git submodule update --init --recursive; then
                        print_success "Repository updated successfully"
                        save_state "repo_cloned"
                        return 0
                    else
                        print_error "Failed to update repository"
                        exit 1
                    fi
                    ;;
                2)
                    print_info "Using existing files, continuing with installation..."
                    cd "$INSTALL_DIR"
                    save_state "repo_cloned"
                    return 0
                    ;;
                3)
                    if [[ "$SCRIPT_IN_BLASPHEMER" == true ]]; then
                        print_warning "This script is running from inside the blasphemer directory"
                        print_info "Copying script to temp location before removing directory..."
                        TEMP_SCRIPT="/tmp/blasphemer_install_$$.sh"
                        cp "${BASH_SOURCE[0]}" "$TEMP_SCRIPT"
                        chmod +x "$TEMP_SCRIPT"
                        print_info "Restarting from temp location..."
                        exec "$TEMP_SCRIPT"
                    fi
                    print_step "Removing existing directory..."
                    rm -rf "$INSTALL_DIR"
                    ;;
                4)
                    print_info "Installation cancelled by user"
                    exit 0
                    ;;
                *)
                    print_error "Invalid option"
                    exit 1
                    ;;
            esac
        else
            # Directory exists but is not a git repo
            print_warning "Directory exists but is not a git repository"
            read -p "Remove and start fresh? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                if [[ "$SCRIPT_IN_BLASPHEMER" == true ]]; then
                    print_warning "This script is running from inside the directory"
                    print_info "Copying script to temp location..."
                    TEMP_SCRIPT="/tmp/blasphemer_install_$$.sh"
                    cp "${BASH_SOURCE[0]}" "$TEMP_SCRIPT"
                    chmod +x "$TEMP_SCRIPT"
                    print_info "Restarting from temp location..."
                    exec "$TEMP_SCRIPT"
                fi
                print_step "Removing existing directory..."
                rm -rf "$INSTALL_DIR"
            else
                print_info "Installation cancelled by user"
                exit 0
            fi
        fi
    fi
    
    # Clone if directory doesn't exist (or was removed above)
    if [[ ! -d "$INSTALL_DIR" ]]; then
        print_step "Cloning repository with submodules..."
        print_info "Repository: $REPO_URL"
        print_info "Destination: $INSTALL_DIR"
        
        if git clone --recursive "$REPO_URL" "$INSTALL_DIR"; then
            print_success "Repository cloned successfully"
            cd "$INSTALL_DIR"
            save_state "repo_cloned"
        else
            print_error "Failed to clone repository"
            exit 1
        fi
    fi
}

################################################################################
# Virtual Environment Setup
################################################################################

create_virtual_environment() {
    if is_step_completed "venv_created"; then
        print_success "Virtual environment already created (resuming installation)"
        return 0
    fi
    
    print_header "Creating Virtual Environment"
    
    cd "$INSTALL_DIR"
    
    print_step "Creating Python virtual environment..."
    if python3 -m venv venv; then
        print_success "Virtual environment created"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
    
    print_step "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    print_step "Upgrading pip..."
    if python -m pip install --upgrade pip > /dev/null 2>&1; then
        print_success "pip upgraded"
    else
        print_warning "Failed to upgrade pip (continuing anyway)"
    fi
    
    save_state "venv_created"
}

################################################################################
# Python Dependencies
################################################################################

install_python_dependencies() {
    if is_step_completed "dependencies_installed"; then
        print_success "Python dependencies already installed (resuming installation)"
        return 0
    fi
    
    print_header "Installing Python Dependencies"
    
    cd "$INSTALL_DIR"
    source venv/bin/activate
    
    print_step "Installing Blasphemer in editable mode..."
    print_info "This may take a few minutes..."
    
    if pip install -e . ; then
        print_success "Blasphemer and dependencies installed"
    else
        print_error "Failed to install dependencies"
        print_info "Check the error messages above for details"
        exit 1
    fi
    
    print_step "Verifying installation..."
    if blasphemer --help > /dev/null 2>&1; then
        print_success "Blasphemer command is working"
    else
        print_error "Blasphemer command failed"
        exit 1
    fi
    
    save_state "dependencies_installed"
}

################################################################################
# llama.cpp Build
################################################################################

build_llama_cpp() {
    if is_step_completed "llama_cpp_built"; then
        print_success "llama.cpp already built (resuming installation)"
        return 0
    fi
    
    print_header "Building llama.cpp"
    
    cd "$INSTALL_DIR/llama.cpp"
    
    print_step "Configuring llama.cpp with cmake..."
    print_info "This will enable Metal support for Apple Silicon GPU acceleration"
    
    if cmake -B build > /dev/null 2>&1; then
        print_success "cmake configuration completed"
    else
        print_error "cmake configuration failed"
        exit 1
    fi
    
    print_step "Building llama-quantize (this may take a few minutes)..."
    local num_cores=$(sysctl -n hw.ncpu)
    print_info "Using $num_cores CPU cores for parallel build"
    
    if cmake --build build --config Release --target llama-quantize -j "$num_cores"; then
        print_success "llama.cpp built successfully"
    else
        print_error "Failed to build llama.cpp"
        exit 1
    fi
    
    print_step "Verifying build..."
    if [[ -f "build/bin/llama-quantize" ]]; then
        print_success "llama-quantize binary created"
    else
        print_error "llama-quantize binary not found"
        exit 1
    fi
    
    cd "$INSTALL_DIR"
    save_state "llama_cpp_built"
}

################################################################################
# Post-Installation
################################################################################

post_install() {
    print_header "Installation Complete!"
    
    echo -e "${GREEN}${BOLD}Blasphemer has been successfully installed!${NC}"
    echo ""
    echo "Installation directory: ${BOLD}$INSTALL_DIR${NC}"
    echo ""
    echo -e "${CYAN}${BOLD}Quick Start:${NC}"
    echo ""
    echo "  1. Activate the virtual environment:"
    echo -e "     ${BOLD}cd $INSTALL_DIR${NC}"
    echo -e "     ${BOLD}source venv/bin/activate${NC}"
    echo ""
    echo "  2. Process your first model:"
    echo -e "     ${BOLD}blasphemer microsoft/Phi-3-mini-4k-instruct${NC}"
    echo ""
    echo "  3. Convert to GGUF for LM Studio:"
    echo -e "     ${BOLD}./convert-to-gguf.sh ~/models/Phi-3-mini-4k-instruct-blasphemer${NC}"
    echo ""
    echo -e "${CYAN}${BOLD}Documentation:${NC}"
    echo ""
    echo -e "  Complete guide: ${BOLD}$INSTALL_DIR/USER_GUIDE.md${NC}"
    echo -e "  Quick commands: ${BOLD}blasphemer --help${NC}"
    echo ""
    echo -e "${CYAN}${BOLD}Features:${NC}"
    echo ""
    echo "  ✓ Apple Silicon MPS GPU support"
    echo "  ✓ Automatic checkpoint/resume system"
    echo "  ✓ LM Studio GGUF conversion"
    echo "  ✓ Professional documentation"
    echo ""
    echo -e "${YELLOW}${BOLD}Tip:${NC} Start with a small model like Phi-3-mini to test the workflow"
    echo ""
    
    # Clean up state file
    rm -f "$STATE_FILE"
    
    print_success "State file cleaned up"
    echo ""
    echo -e "${GREEN}Happy model decensoring! 🚀${NC}"
    echo ""
}

################################################################################
# Error Handler
################################################################################

cleanup_on_error() {
    local exit_code=$?
    echo ""
    print_error "Installation interrupted (exit code: $exit_code)"
    echo ""
    print_info "Your progress has been saved to: $STATE_FILE"
    print_info "Run this script again to resume from where you left off"
    echo ""
    print_info "To start fresh, delete the state file:"
    echo "  rm -f $STATE_FILE"
    echo ""
    exit $exit_code
}

trap cleanup_on_error ERR INT TERM

################################################################################
# Main Installation Flow
################################################################################

main() {
    clear
    
    print_header "Blasphemer Installation for macOS"
    
    echo -e "${CYAN}Developed by Christopher Bradford (@sunkencity999)${NC}"
    echo -e "${CYAN}https://github.com/sunkencity999/blasphemer${NC}"
    echo ""
    echo "This script will install Blasphemer and all its dependencies."
    echo ""
    
    local current_state=$(load_state)
    if [[ "$current_state" != "start" && "$current_state" != "completed" ]]; then
        print_info "Resuming from previous installation attempt"
        print_info "Current state: $current_state"
        echo ""
    fi
    
    # Show installation plan
    echo -e "${BOLD}Installation steps:${NC}"
    echo "  1. Check prerequisites"
    echo "  2. Clone repository"
    echo "  3. Create virtual environment"
    echo "  4. Install Python dependencies"
    echo "  5. Build llama.cpp"
    echo ""
    
    read -p "Press Enter to continue or Ctrl+C to cancel..."
    echo ""
    
    # Run installation steps
    check_prerequisites
    clone_repository
    create_virtual_environment
    install_python_dependencies
    build_llama_cpp
    
    # Mark as completed
    save_state "completed"
    
    post_install
}

# Run main installation
main "$@"
