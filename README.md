# BGE-M3 ONNX Quantized

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) INT8 dynamic quantized ONNX model with pruned tokenizer for Buddhist text semantic search.

## What's Changed

| | Original | Quantized |
|---|---|---|
| Format | PyTorch | ONNX INT8 |
| Model size | ~2.2GB | ~550MB |
| Tokenizer vocab | 250,002 | ~120,000 |
| Embedding dim | 1024 | 1024 |

### Quantization
- INT8 dynamic quantization via `onnxruntime`
- Per-channel weight quantization

### Tokenizer Pruning
Removed unused language tokens, keeping only:
- Latin + diacritics (English, Pali romanization)
- Korean (한글)
- CJK (漢文 Buddhist texts)
- Devanagari (Sanskrit/Pali)

## Usage

### Python (onnxruntime)
```python
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

sess = ort.InferenceSession("model_quantized.onnx")
tok = AutoTokenizer.from_pretrained("./")

def embed(texts):
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in [i.name for i in sess.get_inputs()]:
        inputs["token_type_ids"] = np.zeros_like(enc["input_ids"]).astype(np.int64)
    out = sess.run(None, inputs)[0]
    mask = enc["attention_mask"][..., np.newaxis]
    pooled = (out * mask).sum(1) / mask.sum(1)
    return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

embeddings = embed(["고통의 원인은 무엇인가", "What is the origin of suffering?"])
print(f"similarity: {np.dot(embeddings[0], embeddings[1]):.4f}")
```

### Flutter (on-device)
```dart
// onnxruntime_flutter package
final session = await OrtSession.fromAsset('assets/models/bge-m3/model_quantized.onnx');
```

## Files

| File | Description |
|---|---|
| `model_quantized.onnx` | INT8 quantized ONNX model |
| `tokenizer.json` | Pruned tokenizer (Latin/Korean/CJK/Devanagari only) |
| `tokenizer_config.json` | Tokenizer config |
| `special_tokens_map.json` | Special tokens |
| `config.json` | Model config |

## Reproduce

Run on Google Colab (T4 GPU):

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer

# 1. Export to ONNX
model = ORTModelForFeatureExtraction.from_pretrained("BAAI/bge-m3", export=True)
model.save_pretrained("./onnx")
AutoTokenizer.from_pretrained("BAAI/bge-m3").save_pretrained("./onnx")

# 2. INT8 dynamic quantization
quantize_dynamic(
    model_input="./onnx/model.onnx",
    model_output="./quantized/model_quantized.onnx",
    per_channel=True,
    weight_type=QuantType.QInt8,
)
```

## License

Same as [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (MIT License).

## Acknowledgements

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) - Base model
- [ONNX Runtime](https://onnxruntime.ai/) - Quantization framework
