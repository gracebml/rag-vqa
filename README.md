# RAG-Enhanced Visual Question Answering cho Lịch sử & Văn hóa Việt Nam

Hệ thống VQA (Visual Question Answering) được tăng cường bằng RAG (Retrieval Augmented Generation) để trả lời câu hỏi về lịch sử và văn hóa Việt Nam dựa trên hình ảnh.

## 🎯 Tổng quan

Hệ thống này kết hợp:
- **Qwen2VL-7B**: Vision Language Model để hiểu hình ảnh và tạo câu trả lời
- **RAG**: Retrieval Augmented Generation để bổ sung kiến thức từ database lịch sử
- **OCR**: Đọc văn bản trong hình ảnh (bia đá, câu đối, v.v.)
- **Wikipedia Integration**: Tìm kiếm thông tin bổ sung từ Wikipedia tiếng Việt

## 📁 Cấu trúc thư mục

```
NLP_FinalProject/
├── report.pdf                  # Báo cáo
├── slides.pdf                  # Slide thuyết trình
├── README.md                   # File này
├── data/                       # Dữ liệu
│   ├── vqa_test.json           # File test tập câu hỏi - ảnh
│   └── knowledge_base.json     # File dữ liệu lịch sử đã xử lý
├── models/                     # (Optional) Model files nhỏ
└── code/                       # Source code chính
    ├── requirements.txt        # Các thư viện cần pip install
    ├── app.py                  # File chạy Demo Gradio
    ├── src/                    # Mã nguồn lõi
    │   ├── __init__.py
    │   ├── config.py           # Cấu hình
    │   ├── vision.py           # Module Captioning & OCR
    │   ├── retrieval.py        # Module RAG (Vector DB & Search)
    │   ├── answering.py         # Module Trả lời
    │   └── pipeline.py         # Module kết hợp (Main Logic)
    └── notebooks/              # Các file chạy trên Kaggle
        ├── 1_Build_Index.ipynb # Tạo Vector Database
        └── 2_Evaluate.ipynb    # Chạy đánh giá trên tập Test
```

## 🚀 Cài đặt

### 1. Clone repository và cài đặt dependencies

```bash
cd NLP_FinalProject/code
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

- Đặt file `knowledge_base.json` vào thư mục `data/`
- Đặt file `vqa_test.json` vào thư mục `data/`
- (Optional) Chuẩn bị thư mục ảnh nếu cần test local

### 3. Cấu hình

Chỉnh sửa file `src/config.py` để điều chỉnh:
- Đường dẫn đến dữ liệu
- Model names
- Retrieval method (BM25, Embedding, hoặc Hybrid)
- Generation parameters

## 💻 Sử dụng

### Chạy Demo với Gradio

```bash
cd code
python app.py
```

Sau đó mở trình duyệt tại `http://localhost:7860`

### Sử dụng trong code Python

```python
from src.pipeline import RAGVQAPipeline
from PIL import Image

# Initialize pipeline
pipeline = RAGVQAPipeline(use_4bit=True)

# Load image
image = Image.open("path/to/image.jpg")

# Ask question
question = "Đây là gì? Ý nghĩa lịch sử của hình ảnh này là gì?"

# Get answer
result = pipeline.process(image, question, return_intermediate=True)
print(result["answer"])
```

## 📊 Chạy trên Kaggle

### 1. Build Vector Index

Chạy notebook `notebooks/1_Build_Index.ipynb` trên Kaggle để:
- Tạo embeddings từ knowledge base
- Build FAISS index
- Lưu metadata

**Lưu ý**: 
- Upload `knowledge_base.json` lên Kaggle Dataset
- Điều chỉnh đường dẫn trong notebook
- Output sẽ được lưu trong `/kaggle/working/models/`

### 2. Evaluate

Chạy notebook `notebooks/2_Evaluate.ipynb` để:
- Đánh giá hệ thống trên tập test
- Tính các metrics
- Lưu kết quả

**Lưu ý**:
- Upload `vqa_test.json` và thư mục ảnh lên Kaggle Dataset
- Điều chỉnh đường dẫn trong notebook
- Kết quả sẽ được lưu trong `/kaggle/working/results/`

## 🔧 Pipeline xử lý

Hệ thống hoạt động theo 3 bước chính:

### 1. Vision Module (Vision-to-Text)
- **Captioning**: Sử dụng Qwen2VL để tạo mô tả chi tiết hình ảnh bằng tiếng Việt
- **OCR**: Sử dụng PaddleOCR hoặc Tesseract để đọc văn bản trong ảnh

### 2. Knowledge Retrieval (RAG)
- **Query Generation**: Kết hợp câu hỏi + caption + OCR text
- **Search Methods**:
  - **BM25**: Tìm kiếm dựa trên từ khóa (tốt cho tên riêng, thuật ngữ)
  - **Embedding**: Tìm kiếm semantic (tốt cho ý nghĩa, ngữ cảnh)
  - **Hybrid**: Kết hợp cả hai phương pháp
- **Wikipedia Fallback**: Tìm kiếm Wikipedia tiếng Việt nếu local KB không có kết quả

### 3. Answering Module (VLM)
- Sử dụng Qwen2VL-7B với 4-bit quantization (chạy được trên T4 GPU)
- Prompt bao gồm: caption, OCR, retrieved context, và question
- Generate answer bằng tiếng Việt với giải thích chi tiết

## 📝 Cấu hình

Các tham số chính trong `src/config.py`:

```python
# Models
QWEN2VL_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
VIETNAMESE_EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

# Retrieval
RETRIEVAL_METHOD = "hybrid"  # "bm25", "embedding", or "hybrid"
TOP_K_RETRIEVE = 3

# Generation
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
```

## 🐛 Troubleshooting

### Lỗi Out of Memory (OOM)
- Giảm `MAX_NEW_TOKENS` trong config
- Đảm bảo sử dụng 4-bit quantization (`use_4bit=True`)
- Giảm batch size nếu có

### Lỗi import modules
- Đảm bảo đã cài đặt tất cả dependencies: `pip install -r requirements.txt`
- Kiểm tra Python version (>= 3.8)

### Lỗi load model
- Kiểm tra kết nối internet (để download từ HuggingFace)
- Kiểm tra disk space (model ~14GB)
- Đảm bảo có GPU với đủ VRAM (tối thiểu 16GB cho full model, 8GB cho 4-bit)

## 📚 Tài liệu tham khảo

- [Qwen2VL Model Card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [RAG Framework for VQA](https://towardsdatascience.com/a-simple-framework-for-rag-enhanced-visual-question-answering-06768094762e/)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)

## 👥 Tác giả

NLP Final Project - RAG-Enhanced VQA for Vietnamese History & Culture

## 📄 License

MIT License

