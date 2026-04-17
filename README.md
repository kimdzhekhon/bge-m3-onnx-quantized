<div align="center">

# 🔤 BGE-M3 ONNX Quantized

**다국어 시맨틱 검색을 위한 경량화 임베딩 모델** — INT8 동적 양자화 ONNX 모델

[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/jaehyun-kim/bge-m3-onnx-quantized)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

</div>

---

## 🌟 개요

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)를 한국어 → 불경(영문/팔리어) 크로스링구얼 검색에 최적화한 INT8 동적 양자화 ONNX 모델입니다.

**HuggingFace:** https://huggingface.co/jaehyun-kim/bge-m3-onnx-quantized

## 🛠 최적화 내역

| 항목 | 원본 | 최적화 후 |
|------|------|---------|
| **포맷** | PyTorch | ONNX |
| **모델 크기** | ~2.2 GB | ~550 MB |
| **양자화** | FP32 | INT8 동적 양자화 |
| **토크나이저 어휘** | 250,002 | ~120,000 (프루닝) |
| **임베딩 차원** | 1024d | 1024d |

## 🔍 핵심 기술 상세

### INT8 동적 양자화
추론 시점에 활성화 값을 동적으로 양자화합니다. 정적 양자화보다 단순하지만 칼리브레이션 없이도 2.2GB → 550MB 압축을 달성합니다.

### 크로스링구얼 검색
BGE-M3의 다국어 능력을 활용하여 한국어 질문 → 영문/팔리어 경전 검색 매핑을 수행합니다.

### Flutter 온디바이스 통합
`onnxruntime` 패키지로 Android/iOS 기기에서 서버 없이 직접 임베딩 생성합니다.