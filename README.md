<div align="center">

# BGE-M3 — ONNX INT8 Quantized

한국어·팔리어 크로스링구얼 불경 검색을 위한 경량화 임베딩 모델 (1024차원, ~550MB)

![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow?style=for-the-badge)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-green?style=for-the-badge)
![Android](https://img.shields.io/badge/Android-On--Device-blue?style=for-the-badge)

![Version](https://img.shields.io/badge/version-1.0.0-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Quantization](https://img.shields.io/badge/quantization-INT8-orange?style=flat-square)

**[HuggingFace 모델 보기 →](https://huggingface.co/jaehyun-kim/bge-m3-onnx-quantized)**

</div>

---

## 목차

1. [소개](#소개)
2. [주요 기능](#주요-기능)
3. [기술 스택 / 최적화 내역](#기술-스택--최적화-내역)
4. [아키텍처 / 구현 원리](#아키텍처--구현-원리)
5. [데이터 흐름](#데이터-흐름)
6. [설치 및 사용](#설치-및-사용)
7. [Flutter 통합](#flutter-통합)
8. [Roadmap](#roadmap)
9. [라이선스](#라이선스)

---

## 소개

BAAI/bge-m3를 한국어→영어/팔리어 크로스링구얼 불경 텍스트 검색에 최적화한 ONNX INT8 양자화 모델입니다. 원본 2.2GB FP32 모델을 어휘 프루닝과 Dynamic INT8 양자화로 약 550MB까지 줄여 Android 기기에서 서버 없이 실행할 수 있습니다. 한국어로 질의하면 영어·팔리어 원문 경전을 교차 검색할 수 있어 다국어 불교 텍스트 코퍼스에 특히 유용합니다. 1024차원 임베딩 전체를 활용하여 높은 의미 표현력을 유지합니다.

> 이 모델은 한국어 사용자가 팔리어 니카야 및 영어 불경 번역본을 자연어로 검색할 수 있도록 설계되었습니다.

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| 크로스링구얼 검색 | 한국어 쿼리로 영어·팔리어 문서 검색 지원 |
| Dynamic INT8 양자화 | 칼리브레이션 없이 적용 가능한 런타임 양자화 |
| 어휘 프루닝 | 한국어·영어·팔리어 관련 토큰만 보존하여 모델 경량화 |
| 1024차원 임베딩 | 원본 임베딩 차원 유지로 높은 의미 표현력 보장 |
| 온디바이스 추론 | ONNX Runtime으로 Android에서 서버 없이 실행 |
| Flutter 통합 | onnxruntime Flutter 패키지로 간편하게 연동 |

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 기술 스택 / 최적화 내역

| 항목 | 내용 |
|---|---|
| 베이스 모델 | BAAI/bge-m3 |
| 원본 크기 | ~2.2GB (FP32 PyTorch) |
| 최종 크기 | ~550MB |
| 양자화 방식 | Dynamic INT8 (칼리브레이션 불필요) |
| 변환 도구 | Hugging Face optimum 라이브러리 |
| 어휘 크기 변화 | 250,002 → ~120,000 토큰 |
| 보존 어휘 | 한국어 / 영어 / 팔리어 |
| 임베딩 차원 | 1024d (원본 유지) |
| 추론 런타임 | ONNX Runtime (Android) |
| 검색 방식 | 크로스링구얼 비대칭 검색 |

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 아키텍처 / 구현 원리

```
[BGE-M3 ONNX INT8 아키텍처]

한국어 입력 쿼리
    │
    ▼
┌─────────────────────────────────────┐
│  다국어 토크나이저 (~120k 어휘)      │
│  - 한/영/팔리어 어휘 보존             │
│  - BPE 서브워드 분리                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  ONNX XLM-RoBERTa 인코더 (INT8)     │
│  - Dynamic INT8 양자화 가중치         │
│  - 크로스링구얼 어텐션                │
│  - 24레이어 트랜스포머                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  1024차원 임베딩 출력                 │
│  - L2 정규화                          │
│  - 코사인 유사도 기반 검색            │
└─────────────────────────────────────┘
```

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 데이터 흐름

```
원본 BAAI/bge-m3 (2.2GB, 250k 어휘)
      │
      ▼
어휘 프루닝 (한/영/팔리어 120k 토큰 보존)
      │
      ▼
ONNX 변환
      │
      ▼
Dynamic INT8 양자화 (칼리브레이션 불필요)
      │
      ▼
최종 모델 ~550MB → Android 기기 배포
```

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 설치 및 사용

**요구사항**

- Python 3.8+
- `onnxruntime >= 1.16`
- `transformers >= 4.35`
- `optimum[onnxruntime]`

```bash
# HuggingFace Hub에서 모델 다운로드
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="jaehyun-kim/bge-m3-onnx-quantized",
    local_dir="./model"
)
EOF
```

```python
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./model")
session = ort.InferenceSession("./model/model.onnx")

# 한국어 쿼리 → 영어/팔리어 문서 크로스링구얼 검색
query = "반야바라밀다의 의미는 무엇인가"
inputs = tokenizer(query, return_tensors="np", padding=True, truncation=True, max_length=512)

outputs = session.run(None, {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"])),
})

embedding = outputs[0][0]  # 1024차원 임베딩
```

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## Flutter 통합

Flutter Android 앱에서 ONNX Runtime으로 온디바이스 크로스링구얼 검색을 수행하는 방법입니다.

```yaml
# pubspec.yaml
dependencies:
  onnxruntime: ^1.x.x
```

```dart
import 'package:onnxruntime/onnxruntime.dart';

// 모델 로드
final session = OrtSession.fromFile(
  'assets/model.onnx',
  OrtSessionOptions(),
);

// 한국어 쿼리 토큰화 후 입력 준비
final List<int> tokenIds = /* 토크나이저 출력 */;
final int seqLen = tokenIds.length;

final inputs = {
  'input_ids': OrtValueTensor.createTensorWithDataList(
    tokenIds,
    [1, seqLen],
  ),
  'attention_mask': OrtValueTensor.createTensorWithDataList(
    List.filled(seqLen, 1),
    [1, seqLen],
  ),
};

// 추론 실행
final outputs = session.run(null, inputs);
final embedding = outputs[0]?.value as List<List<double>>;
// embedding[0] → 1024차원 벡터, 코사인 유사도로 팔리어·영어 문서 검색
```

> Android `assets/` 폴더에 `model.onnx`와 토크나이저 파일을 포함시키고, `pubspec.yaml`의 `flutter.assets`에 경로를 등록하세요.

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## Roadmap

- [x] Dynamic INT8 양자화
- [x] 어휘 프루닝 (250k → 120k 토큰)
- [x] ONNX 변환
- [x] 크로스링구얼 검색 (한국어 → 영어/팔리어)
- [x] Flutter 온디바이스 통합
- [ ] Static INT8 버전 추가
- [ ] FP16 버전 추가
- [ ] 벤치마크 문서 (속도·정확도·크로스링구얼 성능)
- [ ] iOS CoreML 포팅

<div align="right"><a href="#목차">↑ 맨 위로</a></div>

---

## 라이선스

MIT License

Copyright (c) 2026 kimdzhekhon

본 저장소의 최적화 코드 및 변환 스크립트는 MIT 라이선스로 배포됩니다. 베이스 모델(BAAI/bge-m3)의 라이선스는 [HuggingFace 모델 카드](https://huggingface.co/BAAI/bge-m3)를 확인하세요.

<div align="right"><a href="#목차">↑ 맨 위로</a></div>
