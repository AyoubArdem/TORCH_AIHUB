# EmbedInvariantPrintDescriptor

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-contrib-5c3ee8.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](./LICENSE)
[![PyPI Version](https://img.shields.io/pypi/v/embed-invariantprintdescriptor?logo=pypi&logoColor=white)](https://pypi.org/project/embed-invariantprintdescriptor/)



`EmbedInvariantPrintDescriptor` is a Python image embedding library that users can install and use with an SDK-style API, similar to an embeddings client.

## Install

```bash
pip install embed-invariantprintdescriptor
```

For local development:

```bash
pip install -e .[dev]
```

## Quick Start

```python
from embed_invariantprintdescriptor import Client

client = Client()

response = client.embeddings.create(
    input="cat.jpg",
    model="embed-invariantprintdescriptor-v1",
)

print(response.object)
print(response.model)
print(response.data[0].embedding.shape)
```

Batch input:

```python
from embed_invariantprintdescriptor import Client

client = Client(device="cpu")
response = client.embeddings.create(input=["cat.jpg", "dog.jpg"])

for item in response.data:
    print(item.index, item.object, item.source, item.embedding.shape)
```

## Functional API

```python
from embed_invariantprintdescriptor import load_model, encode_image

model = load_model()
result = encode_image("cat.jpg", model=model)
print(result.embedding[:8])
```

Batch functional usage:

```python
from embed_invariantprintdescriptor import load_model, encode_images

model = load_model()
results = encode_images(["cat.jpg", "dog.jpg"], model=model)

for item in results:
    print(item.index, item.embedding.shape)
```

## Full Public API

The package currently exports these public objects:

- `Client`
- `EmbedClient`
- `Embedding`
- `EmbeddingResponse`
- `EmbedResult`
- `EmbedInvariantPrintDescriptor`
- `EmbedInvariantPrintDescriptorConfig`
- `load_model`
- `encode_image`
- `encode_images`
- `cosine_similarity`
- `batch_cosine_similarity`

### `Client`

Primary SDK-style entry point.

```python
from embed_invariantprintdescriptor import Client

client = Client()
response = client.embeddings.create(input="image.jpg")
```

### `EmbedClient`

Alias-compatible client class for the same behavior as `Client`.

```python
from embed_invariantprintdescriptor import EmbedClient

client = EmbedClient(device="cpu")
response = client.embeddings.create(input="image.jpg")
```

### `load_model(weights_path=None, device=None, config=None)`

Loads the local PyTorch model and returns an `EmbedInvariantPrintDescriptor` instance.

### `encode_image(image, model=None, model_name="embed-invariantprintdescriptor-v1", weights_path=None, device=None)`

Generates a single embedding item from one image.

### `encode_images(images, model=None, model_name="embed-invariantprintdescriptor-v1", weights_path=None, device=None)`

Generates embedding items from multiple images.

### `Embedding` / `EmbedResult`

Single embedding record with:

- `object`
- `index`
- `embedding`
- `source`
- `keypoint_count`
- `descriptor_count`
- `model`

### `EmbeddingResponse`

Response wrapper with:

- `object`
- `model`
- `data`
- `embeddings`

### `EmbedInvariantPrintDescriptor`

The underlying PyTorch encoder module.

### `EmbedInvariantPrintDescriptorConfig`

Configuration object for the encoder, including descriptor dimension, hidden dimension, head count, output normalization, and noise settings.

### `cosine_similarity(left, right)`

Returns cosine similarity between two embedding vectors.

```python
from embed_invariantprintdescriptor import cosine_similarity

score = cosine_similarity(vec1, vec2)
```

### `batch_cosine_similarity(query, matrix)`

Returns cosine similarity between one query embedding and a matrix of embeddings.

```python
from embed_invariantprintdescriptor import batch_cosine_similarity

scores = batch_cosine_similarity(query, embedding_matrix)
```

## Response Shape

Each `client.embeddings.create(...)` call returns an `EmbeddingResponse`:

```python
response.object
response.model
response.data
response.embeddings
```

Each item in `response.data` contains:

- `object`
- `index`
- `embedding`
- `source`
- `keypoint_count`
- `descriptor_count`
- `model`

