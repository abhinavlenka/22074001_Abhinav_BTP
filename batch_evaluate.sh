#!/bin/bash
# Batch Evaluation Script for English-Hindi MT Pipeline
# Tests different stage combinations and generates comparison report

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <source_file> <reference_file>"
    echo ""
    echo "Example:"
    echo "  $0 data/test.en data/test.hi"
    exit 1
fi

SOURCE_FILE="$1"
REFERENCE_FILE="$2"

# Validate files
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file not found: $SOURCE_FILE"
    exit 1
fi

if [ ! -f "$REFERENCE_FILE" ]; then
    echo "Error: Reference file not found: $REFERENCE_FILE"
    exit 1
fi

# Create output directory
OUTPUT_DIR="evaluation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}Batch Evaluation of English-Hindi MT Pipeline${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "Source file: $SOURCE_FILE"
echo "Reference file: $REFERENCE_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Activate virtual environment
source venv/bin/activate

# Function to extract scores from output
extract_scores() {
    local file=$1
    local bleu=$(grep "BLEU Score:" "$file" | grep -oP '\d+\.\d+' | head -1)
    local char_f2=$(grep "Character F2:" "$file" | grep -oP '\d+\.\d+' | head -1)
    local wer=$(grep "Word Error Rate:" "$file" | grep -oP '\d+\.\d+' | head -1)
    echo "$bleu|$char_f2|$wer"
}

# Stage configurations to test
declare -a CONFIGS=(
    "2:EOLE Only"
    "3:Transducer Only"
    "2 3:EOLE + Transducer"
    "2 3 4:EOLE + Transducer + LLM"
    "1 2 3:SpaCy + EOLE + Transducer"
    "1 2 3 4:Full Pipeline"
)

echo -e "${YELLOW}Running evaluations...${NC}"
echo ""

# Run evaluations
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r stages name <<< "$config"

    echo -e "${GREEN}Testing: $name (stages: $stages)${NC}"

    output_file="$OUTPUT_DIR/eval_stages_$(echo $stages | tr ' ' '_').txt"

    python evaluate_metrics.py \
        --evaluate-pipeline \
        --source "$SOURCE_FILE" \
        --reference "$REFERENCE_FILE" \
        --stages $stages > "$output_file" 2>&1

    echo "  ✓ Results saved to: $output_file"
    echo ""
done

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}Generating Comparison Report${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Generate comparison report
REPORT_FILE="$OUTPUT_DIR/comparison_report.txt"

cat > "$REPORT_FILE" << 'EOF'
======================================================================
ENGLISH-HINDI MT PIPELINE - EVALUATION COMPARISON REPORT
======================================================================

EOF

echo "Generated: $(date)" >> "$REPORT_FILE"
echo "Source: $SOURCE_FILE" >> "$REPORT_FILE"
echo "Reference: $REFERENCE_FILE" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

cat >> "$REPORT_FILE" << 'EOF'
----------------------------------------------------------------------
METRICS COMPARISON
----------------------------------------------------------------------

Configuration                        BLEU    Char F2    WER
-------------------------------------------------------------------
EOF

# Extract and format results
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r stages name <<< "$config"
    output_file="$OUTPUT_DIR/eval_stages_$(echo $stages | tr ' ' '_').txt"

    scores=$(extract_scores "$output_file")
    IFS='|' read -r bleu char_f2 wer <<< "$scores"

    printf "%-36s %6s  %7s%%  %6s%%\n" "$name" "$bleu" "$char_f2" "$wer" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << 'EOF'
-------------------------------------------------------------------

INTERPRETATION GUIDE:
- BLEU Score: 0-100 (higher is better)
  * 70-100: Very good to exceptional
  * 50-70:  Good
  * 30-50:  Understandable
  * 0-30:   Poor

- Character F2: 0-100% (higher is better)
  * 90-100%: Excellent
  * 75-90%:  Good
  * 60-75%:  Moderate
  * <60%:    Poor

- Word Error Rate (WER): 0-∞% (LOWER is better)
  * 0-10%:   Excellent
  * 10-20%:  Good
  * 20-40%:  Moderate
  * 40-60%:  Poor
  * >60%:    Very poor

RECOMMENDATIONS:
1. Compare BLEU scores to identify best-performing configuration
2. Check Char F2 for morphological accuracy
3. Low WER indicates good word-level precision
4. Balance all three metrics for overall quality assessment

EOF

# Display the report
cat "$REPORT_FILE"

# Create markdown version for better viewing
REPORT_MD="$OUTPUT_DIR/comparison_report.md"

cat > "$REPORT_MD" << 'EOF'
# English-Hindi MT Pipeline - Evaluation Report

EOF

echo "**Generated:** $(date)" >> "$REPORT_MD"
echo "**Source:** \`$SOURCE_FILE\`" >> "$REPORT_MD"
echo "**Reference:** \`$REFERENCE_FILE\`" >> "$REPORT_MD"
echo "" >> "$REPORT_MD"
echo "## Metrics Comparison" >> "$REPORT_MD"
echo "" >> "$REPORT_MD"
echo "| Configuration | BLEU | Char F2 | WER |" >> "$REPORT_MD"
echo "|---------------|------|---------|-----|" >> "$REPORT_MD"

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r stages name <<< "$config"
    output_file="$OUTPUT_DIR/eval_stages_$(echo $stages | tr ' ' '_').txt"

    scores=$(extract_scores "$output_file")
    IFS='|' read -r bleu char_f2 wer <<< "$scores"

    echo "| $name | $bleu | ${char_f2}% | ${wer}% |" >> "$REPORT_MD"
done

cat >> "$REPORT_MD" << 'EOF'

## Interpretation Guide

### BLEU Score (0-100, higher is better)
- **70-100**: Very good to exceptional
- **50-70**: Good
- **30-50**: Understandable
- **0-30**: Poor

### Character F2 (0-100%, higher is better)
- **90-100%**: Excellent
- **75-90%**: Good
- **60-75%**: Moderate
- **<60%**: Poor

### Word Error Rate (0-∞%, LOWER is better)
- **0-10%**: Excellent
- **10-20%**: Good
- **20-40%**: Moderate
- **40-60%**: Poor
- **>60%**: Very poor

## Recommendations

1. Compare BLEU scores to identify best-performing configuration
2. Check Char F2 for morphological accuracy
3. Low WER indicates good word-level precision
4. Balance all three metrics for overall quality assessment

## Files Generated

EOF

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r stages name <<< "$config"
    output_file="eval_stages_$(echo $stages | tr ' ' '_').txt"
    echo "- \`$output_file\`: Detailed results for $name" >> "$REPORT_MD"
done

echo "" >> "$REPORT_MD"
echo "---" >> "$REPORT_MD"
echo "*Generated by batch_evaluate.sh*" >> "$REPORT_MD"

# Summary
echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Evaluation Complete!${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Generated files:"
echo "  - comparison_report.txt (plain text)"
echo "  - comparison_report.md (markdown)"
echo "  - eval_stages_*.txt (detailed results for each configuration)"
echo ""
echo "View the report:"
echo "  cat $OUTPUT_DIR/comparison_report.txt"
echo ""
echo "Or open in a text editor:"
echo "  nano $OUTPUT_DIR/comparison_report.md"
echo ""
