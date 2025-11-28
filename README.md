# English-Hindi Machine Translation System

A comprehensive hybrid machine translation system combining transfer-based and neural approaches.

## Architecture

```
Input (English) 
    ↓
[1] SpaCy Syntactic Analysis
    ↓
[2] EOLE Lexical & Syntactic Transfer
    ↓
[3] Neural Transducer Morphological Generation
    ↓
[4] LLM Few-Shot Refinement
    ↓
Output (Hindi)
```

## Components

1. **SpaCy**: Syntactic parsing and feature extraction
2. **EOLE (OpenNMT-py)**: Neural lexical and syntactic transfer
3. **Neural Transducer**: Character-level morphological generation
4. **LLM Refinement**: Few-shot learning with Gemini/GPT APIs

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Optional: Install PyTorch with CUDA for GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Training Pipeline

### Step 1: Prepare Data

```bash
python scripts/01_prepare_data.py
```

This downloads and prepares the IIT Bombay English-Hindi corpus:
- Training data: ~1.5M sentence pairs
- Dev/Test splits: 1K each
- Output: `data/train.en`, `data/train.hi`, etc.

### Step 2: Syntactic Analysis

```bash
python scripts/02_spacy_analysis.py
```

Performs syntactic analysis on English source:
- POS tagging
- Dependency parsing
- Feature extraction
- Output: `data/spacy_features/*.pkl`

### Step 3: Train SentencePiece Models

```bash
python scripts/03_train_sentencepiece.py
```

Trains subword tokenizers:
- English vocabulary: 32K tokens
- Hindi vocabulary: 32K tokens
- Output: `models/spm.en.model`, `models/spm.hi.model`

### Step 4: Train EOLE (NMT) Model

```bash
python scripts/04_train_eole.py --mode train
```

Trains the neural machine translation model:
- 6-layer Transformer encoder/decoder
- Training time: ~24-48 hours on GPU
- Output: `models/eole_model_step_*.pt`

Monitor training:
```bash
tensorboard --logdir logs/tensorboard
```

### Step 5: Prepare Neural Transducer Data

```bash
python scripts/05_prepare_transducer.py
```

Prepares character-level data for morphological transfer:
- Creates aligned character sequences
- Output: `neural-transducer/data/en-hi/`

### Step 6: Train Neural Transducer

```bash
python scripts/06_train_transducer.py
```

Trains character-level morphological model:
- LSTM-based sequence-to-sequence
- Training time: ~4-8 hours on GPU
- Output: `models/transducer/best_model.pt`

### Step 7: Setup LLM API (Optional)

For LLM refinement, set up API keys:

**Google Gemini (Recommended - Free):**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

Get API key: https://makersuite.google.com/app/apikey

**Alternative - OpenAI:**
```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Usage

### Interactive Translation

```bash
python pipeline.py
```

Enter text interactively and see the output from all stages.

### Single Sentence

```bash
python pipeline.py --text "Hello, how are you today?"
```

### File Translation

```bash
python pipeline.py \
    --input_file data/test.en \
    --output_file outputs/test.hi \
    --stages 1 2 3 4
```

### Use Specific Stages

```bash
# Only EOLE and LLM refinement
python pipeline.py --text "Your text" --stages 2 4

# Only SpaCy analysis and EOLE
python pipeline.py --text "Your text" --stages 1 2
```

### Individual Stage Testing

**Test EOLE model:**
```bash
python scripts/04_train_eole.py \
    --mode translate \
    --input data/test.en \
    --output outputs/eole_test.hi \
    --model models/eole_model_step_100000.pt
```

**Test LLM refinement:**
```bash
python scripts/07_llm_refinement.py \
    --english data/test.en \
    --mt_output outputs/eole_test.hi \
    --output outputs/refined_test.hi \
    --api gemini
```

## Evaluation

```bash
# Compute BLEU score
sacrebleu data/test.hi < outputs/test.hi
```

## Project Structure

```
eng-hindi-mt/
├── data/                       # Training data
│   ├── train.en, train.hi     # Training pairs
│   ├── dev.en, dev.hi         # Validation
│   ├── test.en, test.hi       # Test set
│   └── spacy_features/        # Syntactic analyses
├── models/                     # Trained models
│   ├── spm.en.model           # English tokenizer
│   ├── spm.hi.model           # Hindi tokenizer
│   ├── eole_model_*.pt        # EOLE checkpoints
│   └── transducer/            # Transducer model
├── scripts/                    # Training scripts
│   ├── 01_prepare_data.py
│   ├── 02_spacy_analysis.py
│   ├── 03_train_sentencepiece.py
│   ├── 04_train_eole.py
│   ├── 05_prepare_transducer.py
│   ├── 06_train_transducer.py
│   └── 07_llm_refinement.py
├── configs/                    # Configuration files
│   └── eole_config.yml
├── logs/                       # Training logs
├── pipeline.py                 # Main translation pipeline
├── requirements.txt
└── README.md
```

## Customization

### Modify Few-Shot Examples

Edit `scripts/07_llm_refinement.py` and update the `few_shot_examples` list.

### Change Model Architecture

Edit `configs/eole_config.yml` to modify:
- Number of layers
- Hidden dimensions
- Training steps
- Batch size

### Use Different Data

Replace the data loading in `scripts/01_prepare_data.py` with your corpus.

## Troubleshooting

**Out of Memory:**
- Reduce `batch_size` in configs
- Use gradient accumulation (`accum_count`)
- Process data in smaller chunks

**EOLE Installation Issues:**
```bash
pip install eole==0.0.6 --no-deps
pip install -r requirements.txt
```

**CUDA Errors:**
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
```

**API Rate Limits:**
- Increase `--delay` parameter in LLM refinement
- Use batch processing with longer delays

## Performance Expectations

With full training:
- **BLEU Score**: 25-35 
- **Training Time**: 
  - EOLE: 24-48 hours (GPU)
  - Transducer: 4-8 hours (GPU)
- **Inference**: ~0.5-2 seconds per sentence (with LLM)

## References

1. EOLE: https://github.com/eole-nlp/eole
2. Neural Transducer: https://github.com/shijie-wu/neural-transducer
3. IIT Bombay Corpus: https://www.cfilt.iitb.ac.in/iitb_parallel/
4. SpaCy: https://spacy.io/

## License

MIT License - See individual component licenses for dependencies.


