#!/bin/bash
# Master Setup and Training Script for English-Hindi MT System

set -e  # Exit on error

echo "========================================================"
echo "English-Hindi MT System - Complete Setup"
echo "========================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_status "Python version: $python_version"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
print_status "Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_status "Pip upgraded"
echo ""

# Install requirements
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt --use-deprecated=legacy-resolver
print_status "Dependencies installed"
echo ""

# Download SpaCy model
echo "Downloading SpaCy English model..."
python -m spacy download en_core_web_sm
print_status "SpaCy model downloaded"
echo ""

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data models/transducer logs outputs
print_status "Directories created"
echo ""

echo "========================================================"
echo "Setup Complete! Now running training pipeline..."
echo "========================================================"
echo ""

# Function to run training steps
run_step() {
    step_num=$1
    step_name=$2
    step_script=$3
    
    echo ""
    echo "========================================================"
    echo "Step $step_num: $step_name"
    echo "========================================================"
    
    if python "$step_script"; then
        print_status "Step $step_num completed successfully"
    else
        print_error "Step $step_num failed"
        exit 1
    fi
}

# Ask user which steps to run
echo "Which training steps would you like to run?"
echo "1. Data preparation"
echo "2. SpaCy analysis"
echo "3. SentencePiece training"
echo "4. EOLE model training (time-consuming)"
echo "5. Neural Transducer preparation"
echo "6. Neural Transducer training (time-consuming)"
echo "7. All steps"
echo "8. Quick demo (steps 1-3, 5 only)"
echo ""
read -p "Enter your choice (1-8): " choice

case $choice in
    1)
        run_step 1 "Data Preparation" "scripts/01_prepare_data.py"
        ;;
    2)
        run_step 2 "SpaCy Analysis" "scripts/02_spacy_analysis.py"
        ;;
    3)
        run_step 3 "SentencePiece Training" "scripts/03_train_sentencepiece.py"
        ;;
    4)
        print_warning "EOLE training can take 24-48 hours on GPU"
        read -p "Continue? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            run_step 4 "EOLE Training" "scripts/04_train_eole.py"
        fi
        ;;
    5)
        run_step 5 "Transducer Data Preparation" "scripts/05_prepare_transducer.py"
        ;;
    6)
        print_warning "Transducer training can take 4-8 hours on GPU"
        read -p "Continue? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            run_step 6 "Transducer Training" "scripts/06_train_transducer.py"
        fi
        ;;
    7)
        print_warning "Full training can take 30+ hours"
        read -p "Continue? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            run_step 1 "Data Preparation" "scripts/01_prepare_data.py"
            run_step 2 "SpaCy Analysis" "scripts/02_spacy_analysis.py"
            run_step 3 "SentencePiece Training" "scripts/03_train_sentencepiece.py"
            run_step 4 "EOLE Training" "scripts/04_train_eole.py"
            run_step 5 "Transducer Data Preparation" "scripts/05_prepare_transducer.py"
            run_step 6 "Transducer Training" "scripts/06_train_transducer.py"
        fi
        ;;
    8)
        echo "Running quick demo setup..."
        run_step 1 "Data Preparation" "scripts/01_prepare_data.py"
        run_step 2 "SpaCy Analysis" "scripts/02_spacy_analysis.py"
        run_step 3 "SentencePiece Training" "scripts/03_train_sentencepiece.py"
        run_step 5 "Transducer Data Preparation" "scripts/05_prepare_transducer.py"
        
        echo ""
        print_status "Quick demo setup complete!"
        print_warning "Note: Full models not trained. For complete system, run full training."
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "========================================================"
echo "Training Pipeline Complete!"
echo "========================================================"
echo ""
echo "Next steps:"
echo "1. Test the pipeline: python pipeline.py"
echo "2. Translate a file: python pipeline.py --input_file data/test.en --output_file outputs/test.hi"
echo "3. Check README.md for more usage examples"
echo ""

# API setup reminder
echo "========================================================"
echo "Optional: LLM API Setup"
echo "========================================================"
echo ""
echo "For LLM refinement (Stage 4), set up an API key:"
echo ""
echo "Google Gemini (Free):"
echo "  export GEMINI_API_KEY='your_key_here'"
echo "  Get key at: https://makersuite.google.com/app/apikey"
echo ""
echo "OpenAI (Paid):"
echo "  export OPENAI_API_KEY='your_key_here'"
echo ""

print_status "Setup and training script completed successfully!"


