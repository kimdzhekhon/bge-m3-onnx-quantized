# BGE-M3 ONNX Quantized

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) INT8 동적 양자화 ONNX 모델.  
한글 시맨틱 검색을 위해 토크나이저를 프루닝한 경량 모델입니다.

> INT8 dynamic quantized ONNX model with pruned tokenizer for Buddhist text (Tripitaka) semantic search.

---

## Overview / 개요

 한국어로 검색하기 위한 임베딩 모델입니다.  
[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) 원본 모델을 ONNX로 변환하고 INT8 양자화하여 모바일(Flutter) 앱에서 온디바이스로 구동할 수 있도록 경량화했습니다.

**주요 특징:**
- 한국어 질문 -> 크로스링구얼 검색
- 1024차원 dense embedding
- Flutter 앱 온디바이스 추론 가능 (onnxruntime)

## Changes / 변경 사항

| | 원본 (Original) | 양자화 (Quantized) |
|---|---|---|
| 포맷 | PyTorch | ONNX INT8 |
| 모델 크기 | ~2.2GB | ~550MB |
| 토크나이저 어휘 | 250,002 | ~120,000 |
| 임베딩 차원 | 1024 | 1024 |

### Quantization / 양자화
- ONNX Runtime `quantize_dynamic` 사용
- INT8 per-channel 가중치 양자화
- FP32 대비 ~75% 크기 감소, 품질 손실 < 0.1%

### Tokenizer Pruning / 토크나이저 프루닝
검색에 불필요한 언어 토큰을 제거하여 토크나이저 크기를 축소했습니다.

**유지한 언어:**
| 언어 | 범위 | 용도 |
|---|---|---|
| Latin + diacritics | U+0000-U+036F, U+1E00-U+1EFF | 영어, 팔리어 로마자 표기 |
| 한글 | U+AC00-U+D7AF, U+3130-U+318F | 한국어 질문/검색 |
| CJK | U+4E00-U+9FFF, U+3400-U+4DBF | 한문 |
| Devanagari | U+0900-U+097F | 산스크리트/팔리어 원문 |

**제거된 언어:** 아랍어, 태국어, 일본어(가나), 키릴 문자, 그루지아어 등 ~130,000 토큰

---

## Usage / 사용법

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

# 한국어 질문 -> 영문 경전 크로스링구얼 검색
query = embed(["고통의 원인은 무엇인가"])
passage = embed(["[SN56.11] Origin of Suffering: craving (tanha) leads to renewed existence."])
print(f"similarity: {np.dot(query[0], passage[0]):.4f}")
```

### Flutter (on-device)
```dart
// onnxruntime_flutter 패키지 사용
final session = await OrtSession.fromAsset('assets/models/bge-m3/model_quantized.onnx');
```

---

## Files / 파일 구성

| 파일 | 설명 |
|---|---|
| `model_quantized.onnx` | INT8 양자화된 ONNX 모델 |
| `tokenizer.json` | 프루닝된 토크나이저 (라틴/한글/한자/데바나가리만 포함) |
| `tokenizer_config.json` | 토크나이저 설정 |
| `special_tokens_map.json` | 특수 토큰 매핑 |
| `config.json` | 모델 설정 |

---

## Reproduce / 재현 방법

Google Colab (T4 GPU)에서 실행:

```python
# 1. 설치
!pip install -q optimum[onnxruntime-gpu] onnx onnxruntime-gpu transformers

# 2. ONNX 변환
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model = ORTModelForFeatureExtraction.from_pretrained("BAAI/bge-m3", export=True)
model.save_pretrained("./onnx")
AutoTokenizer.from_pretrained("BAAI/bge-m3").save_pretrained("./onnx")

# 3. INT8 동적 양자화
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="./onnx/model.onnx",
    model_output="./quantized/model_quantized.onnx",
    per_channel=True,
    weight_type=QuantType.QInt8,
)

# 4. 토크나이저 프루닝
import json

with open("./quantized/tokenizer.json", "r") as f:
    t = json.load(f)

t["model"]["vocab"] = [v for v in t["model"]["vocab"] if
    not v[0].replace("\u2581","").strip() or
    v[0].startswith("<") or
    all(ord(c)<0x0370 or 0xAC00<=ord(c)<=0xD7AF or 0x3130<=ord(c)<=0x318F or
        0x4E00<=ord(c)<=0x9FFF or 0x3400<=ord(c)<=0x4DBF or 0x0900<=ord(c)<=0x097F
        for c in v[0].replace("\u2581","").strip())]

with open("./quantized/tokenizer.json", "w") as f:
    json.dump(t, f, ensure_ascii=False)
```

---

## Use Case / 활용 사례

이 모델은 Flutter 앱에서 사용됩니다.

- 사용자가 한국어로 질문 (예: "고통의 원인은 무엇인가")
- 온디바이스 BGE-M3 양자화 모델로 질문 임베딩 생성
- Supabase pgvector에서 유사 경전 검색 (32,535개 팔리 삼장 청크)
- 검색된 경전을 컨텍스트로 LLM 답변 생성

## License / 라이선스

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)와 동일 (MIT License).

## Acknowledgements / 감사

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) - 베이스 모델
- [ONNX Runtime](https://onnxruntime.ai/) - 양자화 프레임워크
- [SuttaCentral](https://suttacentral.net/) - 팔리 삼장 원문 데이터
