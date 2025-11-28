#!/bin/bash
# Automated script to download IIT Bombay corpus and run 50k evaluation
# This is the main entry point for large-scale evaluation

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}IIT Bombay English-Hindi MT Evaluation${NC}"
echo -e "${BLUE}50,000 Sentence Test Set${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Run setup.sh first"
    exit 1
fi

source venv/bin/activate

# Check if IIT Bombay corpus is downloaded
if [ ! -f "data/iitb_test_50k.en" ] || [ ! -f "data/iitb_test_50k.hi" ]; then
    echo -e "${YELLOW}IIT Bombay corpus not found. Downloading...${NC}"
    echo ""
    python download_iitb_corpus.py
    echo ""
fi

# Menu for user
echo -e "${GREEN}Select evaluation mode:${NC}"
echo "  1. Quick test (1,000 sentences - ~5 minutes)"
echo "  2. Medium test (5,000 sentences - ~20 minutes)"
echo "  3. Large test (10,000 sentences - ~40 minutes)"
echo "  4. Full evaluation (50,000 sentences - ~3-4 hours)"
echo "  5. Custom configuration"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        SOURCE="data/iitb_test_1000.en"
        REFERENCE="data/iitb_test_1000.hi"
        SIZE="1,000"
        ;;
    2)
        SOURCE="data/iitb_test_5000.en"
        REFERENCE="data/iitb_test_5000.hi"
        SIZE="5,000"
        ;;
    3)
        SOURCE="data/iitb_test_10000.en"
        REFERENCE="data/iitb_test_10000.hi"
        SIZE="10,000"
        ;;
    4)
        SOURCE="data/iitb_test_50k.en"
        REFERENCE="data/iitb_test_50k.hi"
        SIZE="50,000"
        ;;
    5)
        read -p "Enter source file path: " SOURCE
        read -p "Enter reference file path: " REFERENCE
        SIZE="custom"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# Check files exist
if [ ! -f "$SOURCE" ]; then
    echo -e "${RED}Error: Source file not found: $SOURCE${NC}"
    exit 1
fi

if [ ! -f "$REFERENCE" ]; then
    echo -e "${RED}Error: Reference file not found: $REFERENCE${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Source:    $SOURCE"
echo "  Reference: $REFERENCE"
echo "  Size:      $SIZE sentences"
echo ""

# Ask about stages
echo -e "${GREEN}Select pipeline stages:${NC}"
echo "  1. EOLE only (Stage 2)"
echo "  2. EOLE + Transducer (Stages 2, 3)"
echo "  3. EOLE + Transducer + LLM (Stages 2, 3, 4)"
echo "  4. Full pipeline (All stages: 1, 2, 3, 4)"
echo "  5. Custom stages"
echo ""
read -p "Enter your choice (1-5): " stage_choice

case $stage_choice in
    1)
        STAGES="2"
        STAGE_DESC="EOLE only"
        ;;
    2)
        STAGES="2 3"
        STAGE_DESC="EOLE + Transducer"
        ;;
    3)
        STAGES="2 3 4"
        STAGE_DESC="EOLE + Transducer + LLM"
        ;;
    4)
        STAGES="1 2 3 4"
        STAGE_DESC="Full pipeline"
        ;;
    5)
        read -p "Enter stages (space-separated, e.g., '2 3'): " STAGES
        STAGE_DESC="Custom"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}Starting Evaluation${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo "  Dataset:  $SIZE sentences"
echo "  Stages:   $STAGE_DESC ($STAGES)"
echo ""
echo "This will:"
echo "  1. Translate all source sentences"
echo "  2. Calculate BLEU, Character F2, and WER metrics"
echo "  3. Generate detailed report"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Run evaluation
echo ""
echo -e "${GREEN}Running evaluation...${NC}"
echo ""

python evaluate_large_dataset.py \
    --source "$SOURCE" \
    --reference "$REFERENCE" \
    --stages $STAGES

echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Evaluation Complete!${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Results saved to: evaluation_results/"
echo ""
echo "Next steps:"
echo "  • View the report: cat evaluation_results/*/report.txt"
echo "  • View detailed JSON: cat evaluation_results/*/results.json"
echo "  • View translations: cat evaluation_results/*/translations.txt"
echo ""
