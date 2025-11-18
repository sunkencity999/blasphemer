#!/usr/bin/env bash

################################################################################
# Shell Script Tests for Blasphemer
# 
# Validates that all shell scripts are properly formatted and executable
################################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

################################################################################
# Test Functions
################################################################################

print_test_header() {
    echo ""
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

test_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((TESTS_PASSED++))
    ((TESTS_RUN++))
}

test_fail() {
    echo -e "${RED}✗${NC} $1"
    echo -e "  ${RED}Error: $2${NC}"
    ((TESTS_FAILED++))
    ((TESTS_RUN++))
}

################################################################################
# Script Validation Tests
################################################################################

test_script_exists() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    if [[ -f "$script_path" ]]; then
        test_pass "$script_name exists"
        return 0
    else
        test_fail "$script_name exists" "File not found at $script_path"
        return 1
    fi
}

test_script_executable() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    if [[ ! -f "$script_path" ]]; then
        test_fail "$script_name is executable" "File not found"
        return 1
    fi
    
    if [[ -x "$script_path" ]]; then
        test_pass "$script_name is executable"
        return 0
    else
        test_fail "$script_name is executable" "File is not executable (chmod +x needed)"
        return 1
    fi
}

test_script_syntax() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    if [[ ! -f "$script_path" ]]; then
        test_fail "$script_name has valid syntax" "File not found"
        return 1
    fi
    
    if bash -n "$script_path" 2>/dev/null; then
        test_pass "$script_name has valid bash syntax"
        return 0
    else
        local error=$(bash -n "$script_path" 2>&1 || true)
        test_fail "$script_name has valid bash syntax" "$error"
        return 1
    fi
}

test_script_shebang() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    if [[ ! -f "$script_path" ]]; then
        test_fail "$script_name has proper shebang" "File not found"
        return 1
    fi
    
    local first_line=$(head -n 1 "$script_path")
    
    if [[ "$first_line" =~ ^#!/(usr/bin/env\ bash|bin/bash) ]]; then
        test_pass "$script_name has proper shebang"
        return 0
    else
        test_fail "$script_name has proper shebang" "Found: $first_line"
        return 1
    fi
}

test_script_has_help() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    if [[ ! -f "$script_path" ]]; then
        test_fail "$script_name has help functionality" "File not found"
        return 1
    fi
    
    # Check if script contains help or usage information
    if grep -q -E "(--help|-h|Usage:|usage)" "$script_path"; then
        test_pass "$script_name has help/usage information"
        return 0
    else
        test_fail "$script_name has help/usage information" "No help text found"
        return 1
    fi
}

test_script_has_error_handling() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    if [[ ! -f "$script_path" ]]; then
        test_fail "$script_name has error handling" "File not found"
        return 1
    fi
    
    # Check for error handling (set -e or error checks)
    if grep -q -E "(set -e|set -euo|exit 1|\|\| exit)" "$script_path"; then
        test_pass "$script_name has error handling"
        return 0
    else
        test_fail "$script_name has error handling" "No error handling found"
        return 1
    fi
}

test_script_branding() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    if [[ ! -f "$script_path" ]]; then
        test_fail "$script_name has Blasphemer branding" "File not found"
        return 1
    fi
    
    # Check for Blasphemer branding (not Heretic)
    if grep -q "Blasphemer" "$script_path"; then
        # Also check it doesn't have outdated "Heretic" branding (except in attribution)
        if grep -q "Heretic Model" "$script_path" || grep -q "heretic-models" "$script_path"; then
            test_fail "$script_name has updated branding" "Found outdated 'Heretic' references"
            return 1
        else
            test_pass "$script_name has Blasphemer branding"
            return 0
        fi
    else
        test_fail "$script_name has Blasphemer branding" "No Blasphemer branding found"
        return 1
    fi
}

################################################################################
# Integration Tests
################################################################################

test_script_help_works() {
    local script_path="$1"
    local script_name=$(basename "$script_path")
    
    if [[ ! -f "$script_path" || ! -x "$script_path" ]]; then
        test_fail "$script_name --help works" "Script not found or not executable"
        return 1
    fi
    
    # Try running with --help (should exit 0 and produce output)
    if timeout 5s "$script_path" --help &>/dev/null; then
        test_pass "$script_name --help executes successfully"
        return 0
    else
        # Some scripts may not have --help, try -h
        if timeout 5s "$script_path" -h &>/dev/null 2>&1; then
            test_pass "$script_name -h executes successfully"
            return 0
        else
            test_fail "$script_name help flag works" "Neither --help nor -h worked"
            return 1
        fi
    fi
}

################################################################################
# Main Test Execution
################################################################################

main() {
    clear
    print_test_header "Blasphemer Shell Script Tests"
    
    echo "Testing shell scripts in: $PROJECT_ROOT"
    echo ""
    
    # Define scripts to test
    local scripts=(
        "$PROJECT_ROOT/blasphemer.sh"
        "$PROJECT_ROOT/convert-to-gguf.sh"
        "$PROJECT_ROOT/install-macos.sh"
    )
    
    # Run tests for each script
    for script in "${scripts[@]}"; do
        local script_name=$(basename "$script")
        
        echo -e "${BOLD}Testing: $script_name${NC}"
        echo ""
        
        test_script_exists "$script"
        test_script_executable "$script"
        test_script_shebang "$script"
        test_script_syntax "$script"
        test_script_has_help "$script"
        test_script_has_error_handling "$script"
        test_script_branding "$script"
        
        # Integration test (if script exists and is executable)
        if [[ -x "$script" ]]; then
            test_script_help_works "$script"
        fi
        
        echo ""
    done
    
    # Summary
    print_test_header "Test Summary"
    
    echo -e "${BOLD}Tests Run:${NC}    $TESTS_RUN"
    echo -e "${GREEN}${BOLD}Tests Passed:${NC} $TESTS_PASSED"
    
    if [[ $TESTS_FAILED -gt 0 ]]; then
        echo -e "${RED}${BOLD}Tests Failed:${NC} $TESTS_FAILED"
        echo ""
        echo -e "${RED}Some tests failed. Please review the errors above.${NC}"
        exit 1
    else
        echo -e "${GREEN}${BOLD}Tests Failed:${NC} 0"
        echo ""
        echo -e "${GREEN}${BOLD}All tests passed! ✓${NC}"
        exit 0
    fi
}

# Run tests
main "$@"
