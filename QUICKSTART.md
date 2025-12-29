# Quick Start Guide

Hướng dẫn nhanh để chạy hệ thống RAG-Enhanced VQA.

## 1. Cài đặt nhanh

```bash
cd NLP_FinalProject/code
pip install -r requirements.txt
```

## 2. Chuẩn bị dữ liệu

- Copy `knowledge_base.json` vào `NLP_FinalProject/data/`
- Copy `vqa_test.json` vào `NLP_FinalProject/data/`
- (Optional) Copy thư mục ảnh nếu test local

## 3. Chạy Demo

```bash
cd code
python app.py
```

Mở trình duyệt tại `http://localhost:7860`

## 4. Sử dụng trong code

```python
from src.pipeline import RAGVQAPipeline
from PIL import Image

# Initialize
pipeline = RAGVQAPipeline(use_4bit=True)

# Process
image = Image.open("path/to/image.jpg")
question = "Đây là gì?"
result = pipeline.process(image, question)
print(result["answer"])
```

## 5. Chạy trên Kaggle

### Build Index:
1. Upload `knowledge_base.json` lên Kaggle Dataset
2. Chạy `notebooks/1_Build_Index.ipynb`
3. Download output từ `/kaggle/working/models/`

### Evaluate:
1. Upload `vqa_test.json` và ảnh lên Kaggle Dataset
2. Upload code từ `code/src/` lên Kaggle
3. Chạy `notebooks/2_Evaluate.ipynb`
4. Download results từ `/kaggle/working/results/`

## Lưu ý

- Cần GPU với ít nhất 8GB VRAM (cho 4-bit quantization)
- Model sẽ được download từ HuggingFace (~14GB)
- Đảm bảo có kết nối internet để download models

