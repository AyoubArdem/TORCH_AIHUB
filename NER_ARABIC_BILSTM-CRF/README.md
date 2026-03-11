# Arabic Named Entity Recognition (NER) with BiLSTM-CRF

This notebook demonstrates a Named Entity Recognition (NER) system for Arabic text using a Bi-directional LSTM (BiLSTM) network followed by a Conditional Random Field (CRF) layer. The model is trained on the `iahlt/arabic_ner_mafat` dataset from Hugging Face. The notebook also includes examples of using the AI Accelerator (AIAC) CLI tool.

## Table of Contents
1.  [Setup and Installation](#1-setup-and-installation)
2.  [Data Loading and Exploration](#2-data-loading-and-exploration)
3.  [Data Preprocessing](#3-data-preprocessing)
4.  [Model Architecture (BiLSTM-CRF)](#4-model-architecture-bilstm-crf)
5.  [Training the Model](#5-training-the-model)
6.  [Inference and Usage](#6-inference-and-usage)
7.  [AI Accelerator (AIAC) CLI Usage](#7-ai-accelerator-aiac-cli-usage)

---

## 1. Setup and Installation

This section covers installing all necessary Python libraries for the project.

```python
!pip install -q ai-accelerator
%pip install -q torch pytorch-crf
!pip install -q datasets
!pip install -q fasttext
!pip install -q torchinfo
!pip install -q torchtext
```

## 2. Data Loading and Exploration

The `iahlt/arabic_ner_mafat` dataset is loaded using the `datasets` library. The dataset contains Arabic text with named entity annotations.

```python
from datasets import load_dataset
import pandas as pd

ner_ar = load_dataset("iahlt/arabic_ner_mafat")
df = pd.DataFrame(ner_ar["train"])
```

Sample data is displayed to understand its structure, and statistics about the tokens and tags are printed.

```python
df.sample(5)
# ... (and other exploration cells)
```

The `entity_types` dictionary maps IOB-formatted tags to human-readable descriptions:

```python
entity_types = {
    "O": "Outside of a named entity",
    "PER": "People (real or fictional)",
    "ORG": "Organizations, institutions, companies",
    "GPE": "Geo-political entities (countries, cities, states)",
    "LOC": "Non-GPE locations (mountains, rivers, regions)",
    "FAC": "Facilities (buildings, airports, landmarks)",
    "TIMEX": "Time expressions (dates, periods)",
    "TTL": "Titles or professions (President, Prime Minister)",
    "WOA": "Works of art (books, movies, songs)",
    "EVE": "Events (Olympics, World Cup)",
    "DUC": "Products (Apple, BMW, Coca-Cola)",
    "ANG": "Languages (Arabic, French, Hebrew)",
    "INFORMAL": "Informal/slang expressions",
    "MISC": "Miscellaneous (catch-all category)"
}
```

## 3. Data Preprocessing

Tokens and raw tags are extracted, unique words and tags are identified, and mappings from words/tags to numerical indices are created (`word2idx`, `tag2idx`). The data is then prepared into a custom PyTorch `Dataset` to handle variable-length sequences, which is crucial for padding during batching.

```python
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader , TensorDataset, Dataset

# ... (code for creating word2idx, tag2idx)

class CustomNERDataset(Dataset):
    def __init__(self, words_list, tags_list):
        self.words_list = words_list
        self.tags_list = tags_list

    def __len__(self):
        return len(self.words_list)

    def __getitem__(self, idx):
        return self.words_list[idx], self.tags_list[idx]

datasets = CustomNERDataset(all_words_tensors, all_tags_one_hot_tensors)
```

A `collate_fn` is defined to handle padding and packing sequences for the `DataLoader`.

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence , pad_sequence
import os

def collate_fn(batch):
    words_tensors_indexed , tags_hot_encoder = zip(*batch)
    words_tensors_padded = pad_sequence(words_tensors_indexed,batch_first=True)
    tags_hot_encoder_padded = pad_sequence(tags_hot_encoder,batch_first=True)
    length_tensor_words = torch.tensor([len(list_words) for list_words in words_tensors_indexed])
    length_tensor_tags = torch.tensor([len(list_tags) for list_tags in tags_hot_encoder])
    words_packed = pack_padded_sequence(words_tensors_padded, length_tensor_words, batch_first=True, enforce_sorted=False)
    tags_packed = pack_padded_sequence(tags_hot_encoder_padded, length_tensor_tags, batch_first=True, enforce_sorted=False)
    return words_packed , tags_packed

dataloader = DataLoader(datasets, batch_size=32,collate_fn= collate_fn , shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
```

## 4. Model Architecture (BiLSTM-CRF)

The `NER_Arabic` model uses an `nn.Embedding` layer, a `nn.LSTM` (Bi-directional), a `nn.Linear` layer to project LSTM outputs to tag probabilities (emissions), and a `CRF` layer from `torchcrf` to predict the most likely sequence of tags.

```python
import torch.nn as nn
from torchcrf import CRF

class NER_Arabic(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int, device: torch.device):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1 , batch_first=True , bidirectional=True)
        self.linear = nn.Linear(in_features= embedding_dim * 2 ,out_features=num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)
        self.device = device

    def forward(self, input_indices, tags=None):
        embeddings = self.embedding(input_indices)
        output_lstm, _ = self.lstm(embeddings)
        emissions = self.linear(output_lstm)

        if tags is not None:
            return emissions # Return emissions for external loss calculation
        else:
            return self.crf.decode(emissions.to(self.device))
```

## 5. Training the Model

The model is trained using `nn.CrossEntropyLoss` and the Adam optimizer. `torch.amp.autocast` and `GradScaler` are used for mixed-precision training to speed up computation.

```python
import tqdm.auto  as tqdm
from torch.amp import autocast, GradScaler

EPOCHS = 30
embedding_dim = 300

model = NER_Arabic(vocab_size=len(unique_words_list), num_classes=len(unique_tags_list), embedding_dim=embedding_dim, device="cuda").to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scaler = GradScaler()

for epoch in tqdm.tqdm(range(EPOCHS)):
    epoch_loss = 0
    for batch_inputs , batch_labels in dataloader:
        # Unpack and move data to device
        unpacked_batch_inputs, _ = pad_packed_sequence(batch_inputs, batch_first=True)
        unpacked_batch_inputs = unpacked_batch_inputs.to(device)
        unpacked_batch_labels, _ = pad_packed_sequence(batch_labels, batch_first=True)
        unpacked_batch_labels = torch.argmax(unpacked_batch_labels, dim=-1).to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
          emissions = model(unpacked_batch_inputs, tags=unpacked_batch_labels)
          loss = loss_fn(emissions.permute(0, 2, 1), unpacked_batch_labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    if epoch % 5 == 0:
        print(f"\n----------------------------------------------")
        print(f"EPOCH {epoch} ------- LOSS: {avg_loss:.4f}")
        print(f"----------------------------------------------\n")
```

A summary of the model's layers and parameters is also provided.

```python
import torchinfo
summary = torchinfo.summary(model)
summary
```

## 6. Inference and Usage

An `NERARABIC` function is defined to take an Arabic sentence, tokenize it using NLTK's `wordpunct_tokenize`, pass it through the trained model, and print the predicted named entities with their descriptions.

```python
import nltk

def NERARABIC(sentence):
    words = nltk.wordpunct_tokenize(sentence)
    processed_words = [word for word in words if word in word2idx]

    if not processed_words:
        return "No recognizable words in the vocabulary for this sentence."

    words_indices = [word2idx[word] for word in processed_words]
    
    with torch.inference_mode():
        sentence_tensor = torch.tensor(words_indices, dtype=torch.long).unsqueeze(0).to(device)
        decoded_tags = model(sentence_tensor) 
        predicted_tag_indices = decoded_tags[0]

        for word, tag_idx in zip(processed_words, predicted_tag_indices):
            tag = unique_tags_list[tag_idx]
            if tag == 'O':
                base_entity_type = tag
            else:
                base_entity_type = tag.split('-')[-1] 
            
            entity_desc = entity_types.get(base_entity_type, f"Unknown: {base_entity_type}") 
            print(f"{word}: {entity_desc}")

        return ""

# Example usage:
new_sentence = "يزور الرئيس الأمريكي جو بايدن المملكة العربية السعودية هذا الأسبوع"
NERARABIC(new_sentence)
```

## 7. AI Accelerator (AIAC) CLI Usage

This section explores the commands available with the `ai-accelerator` CLI tool, which is designed for deploying, monitoring, and governing AI models.

```python
!aiac --help
# ... (other !aiac commands like server --help, deployment --help, etc.)
```
