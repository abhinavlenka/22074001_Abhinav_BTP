#!/usr/bin/env python3
"""
Simplified Neural Transducer Implementation
Character-level morphological transfer model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np

class CharacterVocab:
    """Build vocabulary from character sequences"""
    def __init__(self):
        self.char2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2char = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.next_idx = 4
    
    def add_char(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = self.next_idx
            self.idx2char[self.next_idx] = char
            self.next_idx += 1
    
    def encode(self, text):
        return [self.char2idx.get(c, self.char2idx['<UNK>']) for c in text]
    
    def decode(self, indices):
        return ''.join([self.idx2char.get(i, '<UNK>') for i in indices])

class TranslationDataset(Dataset):
    """Dataset for character-level translation"""
    def __init__(self, src_file, tgt_file, src_vocab=None, tgt_vocab=None):
        self.src_lines = []
        self.tgt_lines = []

        # Read files line by line to maintain alignment
        with open(src_file, 'r', encoding='utf-8') as src_f, \
             open(tgt_file, 'r', encoding='utf-8') as tgt_f:
            for src_line, tgt_line in zip(src_f, tgt_f):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                # Only include pairs where both are non-empty
                if src_line and tgt_line:
                    self.src_lines.append(src_line)
                    self.tgt_lines.append(tgt_line)

        print(f"Loaded {len(self.src_lines)} parallel sentence pairs")

        # Build vocabularies if not provided
        if src_vocab is None:
            self.src_vocab = CharacterVocab()
            for line in self.src_lines:
                for char in line:
                    self.src_vocab.add_char(char)
        else:
            self.src_vocab = src_vocab

        if tgt_vocab is None:
            self.tgt_vocab = CharacterVocab()
            for line in self.tgt_lines:
                for char in line:
                    self.tgt_vocab.add_char(char)
        else:
            self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.src_lines)
    
    def __getitem__(self, idx):
        # Limit sequence length to reduce memory usage
        max_len = 200
        src_text = self.src_lines[idx][:max_len]
        tgt_text = self.tgt_lines[idx][:max_len]

        src = [self.src_vocab.char2idx['<SOS>']] + self.src_vocab.encode(src_text) + [self.src_vocab.char2idx['<EOS>']]
        tgt = [self.tgt_vocab.char2idx['<SOS>']] + self.tgt_vocab.encode(tgt_text) + [self.tgt_vocab.char2idx['<EOS>']]

        return torch.LongTensor(src), torch.LongTensor(tgt)

def collate_fn(batch):
    """Collate function for variable-length sequences"""
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]
    
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)
    
    src_padded = torch.zeros(len(src_batch), max_src_len, dtype=torch.long)
    tgt_padded = torch.zeros(len(tgt_batch), max_tgt_len, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt
    
    return src_padded, tgt_padded, src_lens, tgt_lens

class Seq2SeqTransducer(nn.Module):
    """Character-level sequence-to-sequence transducer"""
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_dim, padding_idx=0)

        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                               batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim * 2, num_layers,
                               batch_first=True)

        self.output_layer = nn.Linear(hidden_dim * 2, tgt_vocab_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, src, tgt):
        # Encoder
        src_emb = self.dropout(self.src_embedding(src))
        encoder_outputs, (hidden, cell) = self.encoder(src_emb)

        # Reshape bidirectional encoder hidden states for decoder
        # hidden/cell: [num_layers * 2, batch, hidden_dim] -> [num_layers, batch, hidden_dim * 2]
        num_layers = hidden.size(0) // 2
        batch_size = hidden.size(1)
        hidden_dim = hidden.size(2)

        # Combine forward and backward hidden states
        hidden = hidden.view(num_layers, 2, batch_size, hidden_dim)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)

        cell = cell.view(num_layers, 2, batch_size, hidden_dim)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)

        # Decoder
        tgt_emb = self.dropout(self.tgt_embedding(tgt[:, :-1]))  # Exclude last token
        decoder_outputs, _ = self.decoder(tgt_emb, (hidden, cell))

        # Output
        logits = self.output_layer(decoder_outputs)
        return logits

def train_transducer():
    """Train the neural transducer model

    Memory-optimized configuration for GPUs with limited VRAM (3-4 GiB):
    - Batch size: 8 (reduced from 32)
    - Gradient accumulation: 4 steps (effective batch size = 32)
    - Max sequence length: 200 characters
    - Model: embedding=128, hidden=256, layers=1
    """
    print("="*50)
    print("Training Neural Transducer")
    print("="*50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_dir = Path("data")
    model_dir = Path("models/transducer")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading training data...")
    train_dataset = TranslationDataset(
        data_dir / "train.en",
        data_dir / "train.hi"
    )
    
    print("Loading validation data...")
    dev_dataset = TranslationDataset(
        data_dir / "dev.en",
        data_dir / "dev.hi",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab
    )
    
    # Save vocabularies
    with open(model_dir / "vocab.pkl", "wb") as f:
        pickle.dump({
            'src_vocab': train_dataset.src_vocab,
            'tgt_vocab': train_dataset.tgt_vocab
        }, f)
    
    # Data loaders - reduced batch size for memory constraints (3.68 GiB GPU)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                             collate_fn=collate_fn, num_workers=2)
    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False,
                           collate_fn=collate_fn, num_workers=2)
    
    # Model
    print("\nInitializing model...")
    model = Seq2SeqTransducer(
        src_vocab_size=len(train_dataset.src_vocab.char2idx),
        tgt_vocab_size=len(train_dataset.tgt_vocab.char2idx)
    ).to(device)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: ~{num_params * 4 / 1024 / 1024:.2f} MB")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Enable memory optimizations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Training loop
    num_epochs = 1
    best_loss = float('inf')
    accumulation_steps = 4  # Gradient accumulation to simulate batch size of 32

    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (src, tgt, src_lens, tgt_lens) in enumerate(pbar):
            src, tgt = src.to(device), tgt.to(device)

            # Forward pass
            logits = model(src, tgt)

            # Calculate loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss = loss / accumulation_steps  # Normalize for gradient accumulation

            # Backward pass
            loss.backward()

            # Gradient accumulation - only update weights every N steps
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': loss.item() * accumulation_steps})
        
        # Validation
        model.eval()
        dev_loss = 0
        with torch.no_grad():
            for src, tgt, src_lens, tgt_lens in dev_loader:
                src, tgt = src.to(device), tgt.to(device)
                logits = model(src, tgt)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
                dev_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_dev_loss = dev_loss / len(dev_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Dev Loss = {avg_dev_loss:.4f}")
        
        # Save best model
        if avg_dev_loss < best_loss:
            best_loss = avg_dev_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }, model_dir / "best_model.pt")
            print(f"Saved best model with loss: {best_loss:.4f}")
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)

if __name__ == "__main__":
    train_transducer()
