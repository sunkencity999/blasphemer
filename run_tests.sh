#!/usr/bin/env bash

################################################################################
# Blasphemer Test Runner
# 
# Runs all test suites: unit tests, integration tests, and shell script tests
# Developed by Christopher Bradford (@sunkencity999)
# https://github.com/sunkencity999/blasphemer
################################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Test results
ALL_PASSED=true

################################################################################
# Utility Functions
################################################################################

print_banner() {
    clear
    echo -e "${CYAN}${BOLD}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║              Blasphemer Test Suite                             ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

print_header() {
    echo ""
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
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

################################################################################
# Environment Setup
################################################################################

setup_test_environment() {
    print_header "Setting Up Test Environment"
    
    # Check if virtual environment exists
    if [[ ! -d "venv" ]]; then
        print_warning "Virtual environment not found"
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    # Check if pytest is installed
    if ! python -c "import pytest" 2>/dev/null; then
        print_warning "pytest not found"
        print_info "Installing test dependencies..."
        pip install pytest pytest-cov -q
    fi
    
    print_success "Test environment ready"
}

################################################################################
# Test Runners
################################################################################

run_unit_tests() {
    print_header "Running Unit Tests"
    
    if python -m pytest tests/unit/ -v --tb=short; then
        print_success "Unit tests passed"
        return 0
    else
        print_error "Unit tests failed"
        ALL_PASSED=false
        return 1
    fi
}

run_integration_tests() {
    print_header "Running Integration Tests"
    
    if python -m pytest tests/integration/ -v --tb=short; then
        print_success "Integration tests passed"
        return 0
    else
        print_error "Integration tests failed"
        ALL_PASSED=false
        return 1
    fi
}

run_shell_script_tests() {
    print_header "Running Shell Script Tests"
    
    if ./tests/scripts/test_shell_scripts.sh; then
        print_success "Shell script tests passed"
        return 0
    else
        print_error "Shell script tests failed"
        ALL_PASSED=false
        return 1
    fi
}

run_coverage_report() {
    print_header "Generating Coverage Report"
    
    print_info "Running tests with coverage..."
    
    if python -m pytest tests/unit/ tests/integration/ \
        --cov=src/heretic \
        --cov-report=term-missing \
        --cov-report=html \
        --tb=short \
        -q; then
        
        print_success "Coverage report generated"
        print_info "HTML report: htmlcov/index.html"
        return 0
    else
        print_warning "Coverage report generation had issues"
        return 1
    fi
}

################################################################################
# Main Execution
################################################################################

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run Blasphemer test suites"
    echo ""
    echo "Options:"
    echo "  --unit           Run only unit tests"
    echo "  --integration    Run only integration tests"
    echo "  --scripts        Run only shell script tests"
    echo "  --coverage       Run tests with coverage report"
    echo "  --all            Run all tests (default)"
    echo "  --help, -h       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # Run all tests"
    echo "  $0 --unit          # Run only unit tests"
    echo "  $0 --coverage      # Run with coverage"
    echo ""
}

main() {
    local run_mode="${1:-all}"
    
    # Handle help flag
    if [[ "$run_mode" == "--help" || "$run_mode" == "-h" ]]; then
        show_help
        exit 0
    fi
    
    print_banner
    
    # Setup environment
    setup_test_environment
    
    # Run tests based on mode
    case "$run_mode" in
        --unit)
            run_unit_tests
            ;;
        --integration)
            run_integration_tests
            ;;
        --scripts)
            run_shell_script_tests
            ;;
        --coverage)
            run_coverage_report
            ;;
        --all|*)
            run_unit_tests || true
            run_integration_tests || true
            run_shell_script_tests || true
            ;;
    esac
    
    # Summary
    print_header "Test Summary"
    
    if [[ "$ALL_PASSED" == true ]]; then
        echo -e "${GREEN}${BOLD}All tests passed! ✓${NC}"
        echo ""
        echo "Your changes are ready to commit."
        exit 0
    else
        echo -e "${RED}${BOLD}Some tests failed ✗${NC}"
        echo ""
        echo "Please review the errors above and fix the issues."
        echo ""
        echo -e "${YELLOW}Tip:${NC} Run individual test suites to isolate failures:"
        echo "  $0 --unit"
        echo "  $0 --integration"
        echo "  $0 --scripts"
        exit 1
    fi
}

# Run main
main "$@"
