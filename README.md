# RAG-Enhanced Visual Question Answering cho Lịch sử & Văn hóa Việt Nam

Hệ thống VQA (Visual Question Answering) được tăng cường bằng RAG (Retrieval Augmented Generation) để trả lời câu hỏi về lịch sử và văn hóa Việt Nam dựa trên hình ảnh.

##  Tổng quan

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
        ├── 2_Evaluate.ipynb    # Chạy đánh giá trên tập Test
        └── 4_FineTune_VQA.ipynb # Fine-tune VLM cho answer generation
    ├── scripts/                # Utility scripts
    │   └── convert_vqa_to_llamafactory.py  # Convert VQA data cho fine-tuning
    └── finetuning/             # Fine-tuning setup
        ├── llamafactory_config.yaml  # Config cho LLaMA Factory
        ├── dataset_info.json    # Dataset info
        ├── freeze_vision_encoder.py  # Helper để freeze vision encoder
        ├── README_FINETUNING.md # Hướng dẫn chi tiết
        └── QUICKSTART_FINETUNING.md # Quick start guide
```

## Cài đặt

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

## Sử dụng

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

## Chạy trên Kaggle

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

## Pipeline xử lý

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

##  Cấu hình

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

## Troubleshooting

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

## Demo resuls:
```
================================================================================
Testing with image: 004160.jpg
Question: Địa điểm trong ảnh có phải di sản thiên nhiên thế giới không?

================================================================================
 DETAILED RESULTS
================================================================================

 CAPTION:
Hình ảnh này là một bức ảnh chụp từ trên cao, cho thấy một cảnh quan tự nhiên tuyệt đẹp với hàng ngàn hòn đảo nhỏ và lớn trải dài trên mặt nước. Đây là một khu vực có nhiều hòn đảo và núi đá vôi, tạo nên một cảnh quan độc đáo và huyền ảo. 

Cảnh quan này có nhiều màu sắc đa dạng, từ màu xanh lá cây của cây cỏ trên các hòn đảo, đến màu xanh ngắt của nước biển và màu trắng của các ngôi nhà nhỏ trên các hòn đảo. Có rất nhiều thuyền và tàu thuyền đang di chuyển trên mặt nước, tạo nên một cảnh quan sinh động và sôi động.

Cảnh quan này có thể là một phần của Vịnh Hạ Long, một di sản thế giới nổi tiếng ở Việt Nam. Vịnh Hạ Long nổi tiếng với hàng ngàn hòn đảo và núi đá vôi, tạo nên một cảnh quan tự nhiên tuyệt đẹp và độc đáo.

 OCR:

 RETRIEVAL SUMMARY:
  Total retrieved documents: 3
  - From Knowledge Base: 0
  - From Wikipedia: 3

 WIKIPEDIA SEARCH KEYWORDS (VLM-generated):
  Vịnh Hạ Long, di sản thiên nhiên thế giới, Việt Nam, hòn đảo, núi đá vôi.

 KNOWLEDGE BASE PASSAGES: None

================================================================================
 WIKIPEDIA PASSAGES (3):
================================================================================

[1] Title: Vịnh Hạ Long
    URL: https://vi.wikipedia.org/wiki/V%E1%BB%8Bnh_H%E1%BA%A1_Long
    Search Keywords Used: Vịnh Hạ Long, di sản thiên nhiên thế giới, Việt Nam, hòn đảo, núi đá vôi.
    Content:
    Vùng di sản trên vịnh Hạ Long được thế giới công nhận (vùng lõi) có diện tích 434 km², như một hình tam giác với ba đỉnh là đảo Đầu Gỗ (phía Tây), hồ Ba Hầm (phía Nam) và đảo Cống Tây (phía Đông), bao gồm 775 đảo với nhiều hang động, bãi tắm. Vùng kế bên (vùng đệm), là di tích danh thắng quốc gia đã được bộ Văn hóa Thông tin Việt Nam xếp hạng từ năm 1962. Địa hình Hạ Long là đảo, núi xen kẽ giữa các trũng biển, là vùng đất mặn có sú vẹt mọc và những đảo đá vôi vách đứng tạo nên những vẻ đẹp tương phản, kết hợp hài hòa, sinh động các yếu tố: đá, nước và bầu trời.

=== Biển và đảo ===
Các đảo ở vịnh Hạ Long có hai dạng là đảo đá vôi và đảo phiến thạch, tập trung ở hai vùng chính là vùng phía Đông Nam vịnh Bái Tử Long và vùng phía Tây Nam vịnh Hạ Long. Theo thống kê của ban quản lý vịnh Hạ Long, trong tổng số 1.969 đảo của vịnh Hạ Long có đến 1.921 đảo đá với nhiều đảo có độ cao khoảng 200 m. Đây là hình ảnh cổ xưa nhất của địa hình có tuổi kiến tạo địa chất từ 250-280 triệu năm về trước, là kết quả của quá trình vận động nâng lên, hạ xuống nhiều lần từ lục địa thành trũng biển. Quá trình carxtơ bào mòn, phong hóa gần như hoàn toàn tạo ra một vịnh Hạ Long độc nhất vô nhị, với hàng ngàn đảo đá nhiều hình thù, dáng vẻ khác nhau lô nhô trên mặt biển, trong một diện tích không lớn của vùng vịnh.
Vùng tập trung các đảo đá có phong cảnh ngoạn mục và nhiều hang động đẹp là vùng trung tâm Di sản Thiên nhiên vịnh Hạ Long, bao gồm phần lớn vịnh Hạ Long (vùng lõi), một phần vịnh Bái Tử Long và vịnh Lan Hạ thuộc quần đảo Cát Bà (vùng đệm).
Các đảo trên vịnh Hạ Long có những hình thù riêng, không giống bất kỳ hòn đảo nào ven biển Việt Nam và không đảo nào giống đảo nào. Có chỗ đảo quần tụ lại nhìn xa ngỡ chồng chất lên nhau, nhưng cũng có chỗ đảo đứng dọc ngang xen kẽ nhau, tạo thành tuyến chạy dài hàng chục kilômét như một bức tường thành. Đó là một thế giới sinh linh ẩn hiện trong những hình hài bằng đá đã được huyền thoại hóa. Đảo thì giống khuôn mặt ai đó đang hướng về đất liền (hòn Đầu Người); đảo thì giống như một con rồng đang bay lượn trên mặt nước (hòn Rồng); đảo thì lại giống như một ông lão đang ngồi câu cá (hòn Lã Vọng); phía xa là hai cánh buồm nâu đang rẽ sóng nước ra khơi (hòn Cánh Buồm); đảo lại lúp xúp như mâm xôi cúng (hòn Mâm Xôi); rồi hai con gà đang âu yếm vờn nhau trên sóng nước (hòn Trống Mái); đứng giữa biển nước bao la một lư hương khổng lồ như một vật cúng tế trời đất (hòn Lư Hương); đảo khác tựa như nhà sư đứng giữa mặt vịnh bao la chắp tay niệm Phật (hòn Ông Sư); đảo lại có hình tròn cao khoảng 40m trông như chiếc đũa phơi mình trước thiên nhiên (hòn Đũa), mà nhìn từ hướng khác lại giống như vị quan triều đình áo xanh, mũ cánh chuồn, nên dân chài còn gọi là hòn Ông v.v.
Bên cạnh các đảo được đặt tên căn cứ vào hình dáng, là các đảo đặt tên theo sự tích dân gian (núi Bài Thơ, hang Trinh Nữ, đảo Tuần Châu), hoặc căn cứ vào các đặc sản có trên đảo hay vùng biển quanh đảo (hòn Ngọc Vừng, hòn Kiến Vàng, đảo Khỉ v.v.). Dưới đây là một vài hòn đảo nổi tiếng:

[2] Title: Danh sách di sản thế giới tại Việt Nam
    URL: https://vi.wikipedia.org/wiki/Danh_s%C3%A1ch_di_s%E1%BA%A3n_th%E1%BA%BF_gi%E1%BB%9Bi_t%E1%BA%A1i_Vi%E1%BB%87t_Nam
    Search Keywords Used: Vịnh Hạ Long, di sản thiên nhiên thế giới, Việt Nam, hòn đảo, núi đá vôi.
    Content:
    Những Di sản thế giới của Tổ chức Giáo dục, Khoa học và Văn hóa Liên Hợp Quốc (UNESCO) là di chỉ, di tích hay danh thắng của một quốc gia được công nhận và quản lý bởi UNESCO. Di sản thế giới tại Việt Nam đã được UNESCO công nhận có đủ cả ba loại hình: di sản thiên nhiên thế giới, di sản văn hóa thế giới và di sản hỗn hợp văn hóa và thiên nhiên thế giới. Trong hệ thống các danh hiệu của UNESCO, di sản thế giới là danh hiệu danh giá nhất và lâu đời nhất.
Các tiêu chí của di sản bao gồm tiêu chí của di sản văn hóa (bao gồm i, ii, iii, iv, v, vi) và di sản thiên nhiên (vii, viii, ix, x). Từ năm 2025, Việt Nam có 9 di sản thế giới được UNESCO công nhận. 6 trong số đó là di sản văn hoá, 2 là di sản tự nhiên và 1 di sản thế giới hỗn hợp. Vườn quốc gia Phong Nha-Kẻ Bàng, Vịnh Hạ Long là những di sản thiên nhiên. Quần thể di tích Cố đô Huế, Phố cổ Hội An, Thánh địa Mỹ Sơn, Khu di tích trung tâm Hoàng thành Thăng Long, Thành nhà Hồ, Quần thể danh thắng Yên Tử - Vĩnh Nghiêm - Côn Sơn, Kiếp Bạc là những di sản văn hoá. Quần thể danh thắng Tràng An là Di sản hỗn hợp duy nhất tại Việt Nam và Đông Nam Á, và là một trong số ít 40 di sản hỗn hợp được UNESCO công nhận.

== Đặc điểm các di sản Việt Nam ==
Các di sản thế giới hiện đều nằm ở nửa phía Bắc của Việt Nam, từ Đà Nẵng trở ra.
Di sản thế giới đáp ứng nhiều tiêu chuẩn nhất: Quần thể danh thắng Tràng An, Vườn Quốc Gia Phong Nha - Kẻ Bàng, Hoàng thành Thăng Long đều đáp ứng 3 tiêu chuẩn.
Di sản thế giới chỉ đáp ứng 1 tiêu chuẩn: Quần thể di tích Cố đô Huế. Có 5 di sản khác đáp ứng 2 tiêu chuẩn.
Các di sản thiên nhiên thế giới (Vịnh Hạ Long-Quần đảo Cát Bà, Quần thể danh thắng Tràng An, VQG Phong Nha - Kẻ Bàng) đều liên quan đến các vùng núi và hang động karst (đáp ứng tiêu chuẩn VIII).
Các di sản văn hóa thế giới (Hoàng thành Thăng Long, Quần thể danh thắng Tràng An, Thành nhà Hồ, Quần thể di tích Cố đô Huế) đều liên quan đến các kinh đô cổ của Việt Nam.
Quần thể danh thắng Tràng An là di sản thế giới hỗn hợp duy nhất ở Việt Nam và Đông Nam Á.
Trong số 7 di sản văn hóa thế giới: có 4/5 di sản đáp ứng tiêu chuẩn II; 3/5 di sản đáp ứng tiêu chuẩn IV; 2/5 di sản đáp ứng tiêu chuẩn V; 3/5 di sản đáp ứng tiêu chuẩn III, 1 di sản đáp ứng tiêu chuẩn VI và chưa có di sản nào đáp ứng tiêu chuẩn I.
Cả ba di sản liên quan đến thiên nhiên đều đáp ứng tiêu chuẩn VIII còn các tiêu chuẩn VII, IX, X chỉ có ở một trong 3 di sản trên.
Di sản Vườn quốc gia Phong Nha - Kẻ Bàng là di sản thiên nhiên liên biên giới đầu tiên của Việt Nam.

[3] Title: Việt Nam
    URL: https://vi.wikipedia.org/wiki/Vi%E1%BB%87t_Nam
    Search Keywords Used: Vịnh Hạ Long, di sản thiên nhiên thế giới, Việt Nam, hòn đảo, núi đá vôi.
    Content:
    Việt Nam có diện tích 331.212 km², đường biên giới trên đất liền dài 4.639 km, đường bờ biển trải dài 3.260 km, có chung đường biên giới trên biển với Thái Lan qua vịnh Thái Lan và với Trung Quốc, Philippines, Indonesia, Brunei, Malaysia qua Biển Đông. Việt Nam tuyên bố chủ quyền đối với hai thực thể địa lí tranh chấp trên Biển Đông là các quần đảo Hoàng Sa (bị mất kiểm soát trên thực tế) và Trường Sa (kiểm soát một phần).
Khoảng cách giữa cực Bắc và cực Nam của Việt Nam theo đường chim bay là 1.650 km. Nơi có chiều ngang hẹp nhất ở Quảng Bình với chưa đầy 50 km. Đường biên giới đất liền dài hơn 4.600 km, trong đó, biên giới với Lào dài nhất (gần 2.100 km), tiếp đến là Trung Quốc và Campuchia. Tổng diện tích là 331.212 km² gồm toàn bộ phần đất liền và hải đảo cùng hơn 4.000 hòn đảo, bãi đá ngầm và cả hai quần đảo trên Biển Đông là Trường Sa (thuộc tỉnh Khánh Hòa) và Hoàng Sa (thuộc thành phố Đà Nẵng) mà nhà nước tuyên bố chủ quyền.
Địa hình Việt Nam có núi rừng chiếm khoảng 40%, đồi 40% và độ che phủ khoảng 75% diện tích đất nước. Có các dãy núi và cao nguyên như dãy Hoàng Liên Sơn, cao nguyên Sơn La ở phía bắc, dãy Bạch Mã và các cao nguyên theo dãy Trường Sơn ở phía nam. Mạng lưới sông, hồ ở vùng đồng bằng châu thổ hoặc miền núi phía Bắc và Tây Nguyên. Đồng bằng chiếm khoảng 1/4 diện tích, gồm các đồng bằng châu thổ như đồng bằng sông Hồng, sông Cửu Long và các vùng đồng bằng ven biển miền Trung, là vùng tập trung dân cư. Đất canh tác chiếm 17% tổng diện tích đất Việt Nam.
Đất chủ yếu là đất ferralit vùng đồi núi (ở Tây Nguyên hình thành trên đá bazan) và đất phù sa đồng bằng. Ven biển đồng bằng sông Hồng và sông Cửu Long tập trung đất phèn. Rừng ở Việt Nam chủ yếu là rừng rậm nhiệt đới khu vực đồi núi còn vùng đất thấp ven biển có rừng ngập mặn. Đất liền có các mỏ khoáng sản như phosphat, vàng. Than đá có nhiều nhất ở Quảng Ninh. Sắt ở Thái Nguyên, Hà Tĩnh. Ở biển có các mỏ dầu và khí tự nhiên.

Số lượng khách du lịch đến Việt Nam tăng nhanh nhất trong vòng 10 năm từ 2000–2010. Năm 2013, có gần 7,6 triệu lượt khách quốc tế đến Việt Nam và năm 2017, có hơn 10 triệu lượt khách quốc tế đến Việt Nam, các thị trường lớn nhất là Trung Quốc, Hàn Quốc, Nhật Bản, Hoa Kỳ và Đài Loan.
Việt Nam có các điểm du lịch từ Bắc đến Nam, từ miền núi tới đồng bằng, từ các thắng cảnh thiên nhiên tới các di tích văn hóa lịch sử. Các điểm du lịch miền núi như Sa Pa, Bà Nà, Đà Lạt. Các điểm du lịch ở các bãi biển như Đà Nẵng, Nha Trang, Vũng Tàu và các đảo như Cát Bà, Côn Đảo, Lý Sơn.

================================================================================
 FINAL ANSWER:
================================================================================
Địa điểm trong ảnh là Vịnh Hạ Long, một di sản thiên nhiên thế giới. Vịnh Hạ Long được UNESCO công nhận là di sản thiên nhiên thế giới vào năm 1994. Đây là một khu vực có hàng ngàn hòn đảo và núi đá vôi, tạo nên một cảnh quan tự nhiên tuyệt đẹp và độc đáo. Vịnh Hạ Long không chỉ có giá trị tự nhiên mà còn có giá trị văn hóa, với nhiều di tích lịch sử và văn hóa được bảo tồn trong khu vực.

================================================================================
 FORMATTED CONTEXT (used for answer generation):
================================================================================
[1] Từ Wikipedia - Vịnh Hạ Long:
Vùng di sản trên vịnh Hạ Long được thế giới công nhận (vùng lõi) có diện tích 434 km², như một hình tam giác với ba đỉnh là đảo Đầu Gỗ (phía Tây), hồ Ba Hầm (phía Nam) và đảo Cống Tây (phía Đông), bao gồm 775 đảo với nhiều hang động, bãi tắm. Vùng kế bên (vùng đệm), là di tích danh thắng quốc gia đã được bộ Văn hóa Thông tin Việt Nam xếp hạng từ năm 1962. Địa hình Hạ Long là đảo, núi xen kẽ giữa các trũng biển, là vùng đất mặn có sú vẹt mọc và những đảo đá vôi vách đứng tạo nên những vẻ đẹp tương phản, kết hợp hài hòa, sinh động các yếu tố: đá, nước và bầu trời.

=== Biển và đảo ===
Các đảo ở vịnh Hạ Long có hai dạng là đảo đá vôi và đảo phiến thạch, tập trung ở hai vùng chính là vùng phía Đông Nam vịnh Bái Tử Long và vùng phía Tây Nam vịnh Hạ Long. Theo thống kê của ban quản lý vịnh Hạ Long, trong tổng số 1.969 đảo của vịnh Hạ Long có đến 1.921 đảo đá với nhiều đảo có độ cao khoảng 200 m. Đây là hình ảnh cổ xưa nhất của địa hình có tuổi kiến tạo địa chất từ 250-280 triệu năm về trước, là kết quả của quá trình vận động nâng lên, hạ xuống nhiều lần từ lục địa thành trũng biển. Quá trình carxtơ bào mòn, phong hóa gần như hoàn toàn tạo ra một vịnh Hạ Long độc nhất vô nhị, với hàng ngàn đảo đá nhiều hình thù, dáng vẻ khác nhau lô nhô trên mặt biển, trong một diện tích không lớn của vùng vịnh.
Vùng tập trung các đảo đá có phong cảnh ngoạn mục và nhiều hang động đẹp là vùng trung tâm Di sản Thiên nhiên vịnh Hạ Long, bao gồm phần lớn vịnh Hạ Long (vùng lõi), một phần vịnh Bái Tử Long và vịnh Lan Hạ thuộc quần đảo Cát Bà (vùng đệm).
Các đảo trên vịnh Hạ Long có những hình thù riêng, không giống bất kỳ hòn đảo nào ven biển Việt Nam và không đảo nào giống đảo nào. Có chỗ đảo quần tụ lại nhìn xa ngỡ chồng chất lên nhau, nhưng cũng có chỗ đảo đứng dọc ngang xen kẽ nhau, tạo thành tuyến chạy dài hàng chục kilômét như một bức tường thành. Đó là một thế giới sinh linh ẩn hiện trong những hình hài bằng đá đã được huyền thoại hóa. Đảo thì giống khuôn mặt ai đó đang hướng về đất liền (hòn Đầu Người); đảo thì giống như một con rồng đang bay lượn trên mặt nước (hòn Rồng); đảo thì lại giống như một ông lão đang ngồi câu cá (hòn Lã Vọng); phía xa là hai cánh buồm nâu đang rẽ sóng nước ra khơi (hòn Cánh Buồm); đảo lại lúp xúp như mâm xôi cúng (hòn Mâm Xôi); rồi hai con gà đang âu yếm vờn nhau trên sóng nước (hòn Trống Mái); đứng giữa biển nước bao la một lư hương khổng lồ như một vật cúng tế trời đất (hòn Lư Hương); đảo khác tựa như nhà sư đứng giữa mặt vịnh bao la chắp tay niệm Phật (hòn Ông Sư); đảo lại có hình tròn cao khoảng 40m trông như chiếc đũa phơi mình trước thiên nhiên (hòn Đũa), mà nhìn từ hướng khác lại giống như vị quan triều đình áo xanh, mũ cánh chuồn, nên dân chài còn gọi là hòn Ông v.v.
Bên cạnh các đảo được đặt tên căn cứ vào hình dáng, là các đảo đặt tên theo sự tích dân gian (núi Bài Thơ, hang Trinh Nữ, đảo Tuần Châu), hoặc căn cứ vào các đặc sản có trên đảo hay vùng biển quanh đảo (hòn Ngọc Vừng, hòn Kiến Vàng, đảo Khỉ v.v.). Dưới đây là một vài hòn đảo nổi tiếng:

[2] Từ Wikipedia - Danh sách di sản thế giới tại Việt Nam:
Những Di sản thế giới của Tổ chức Giáo dục, Khoa học và Văn hóa Liên Hợp Quốc (UNESCO) là di chỉ, di tích hay danh thắng của một quốc gia được công nhận và quản lý bởi UNESCO. Di sản thế giới tại Việt Nam đã được UNESCO công nhận có đủ cả ba loại hình: di sản thiên nhiên thế giới, di sản văn hóa thế giới và di sản hỗn hợp văn hóa và thiên nhiên thế giới. Trong hệ thống các danh hiệu của UNESCO, di sản thế giới là danh hiệu danh giá nhất và lâu đời nhất.
Các tiêu chí của di sản bao gồm tiêu chí của di sản văn hóa (bao gồm i, ii, iii, iv, v, vi) và di sản thiên nhiên (vii, viii, ix, x). Từ năm 2025, Việt Nam có 9 di sản thế giới được UNESCO công nhận. 6 trong số đó là di sản văn hoá, 2 là di sản tự nhiên và 1 di sản thế giới hỗn hợp. Vườn quốc gia Phong Nha-Kẻ Bàng, Vịnh Hạ Long là những di sản thiên nhiên. Quần thể di tích Cố đô Huế, Phố cổ Hội An, Thánh địa Mỹ Sơn, Khu di tích trung tâm Hoàng thành Thăng Long, Thành nhà Hồ, Quần thể danh thắng Yên Tử - Vĩnh Nghiêm - Côn Sơn, Kiếp Bạc là những di sản văn hoá. Quần thể danh thắng Tràng An là Di sản hỗn hợp duy nhất tại Việt Nam và Đông Nam Á, và là một trong số ít 40 di sản hỗn hợp được UNESCO công nhận.

== Đặc điểm các di sản Việt Nam ==
Các di sản thế giới hiện đều nằm ở nửa phía Bắc của Việt Nam, từ Đà Nẵng trở ra.
Di sản thế giới đáp ứng nhiều tiêu chuẩn nhất: Quần thể danh thắng Tràng An, Vườn Quốc Gia Phong Nha - Kẻ Bàng, Hoàng thành Thăng Long đều đáp ứng 3 tiêu chuẩn.
Di sản thế giới chỉ đáp ứng 1 tiêu chuẩn: Quần thể di tích Cố đô Huế. Có 5 di sản khác đáp ứng 2 tiêu chuẩn.
Các di sản thiên nhiên thế giới (Vịnh Hạ Long-Quần đảo Cát Bà, Quần thể danh thắng Tràng An, VQG Phong Nha - Kẻ Bàng) đều liên quan đến các vùng núi và hang động karst (đáp ứng tiêu chuẩn VIII).
Các di sản văn hóa thế giới (Hoàng thành Thăng Long, Quần thể danh thắng Tràng An, Thành nhà Hồ, Quần thể di tích Cố đô Huế) đều liên quan đến các kinh đô cổ của Việt Nam.
Quần thể danh thắng Tràng An là di sản thế giới hỗn hợp duy nhất ở Việt Nam và Đông Nam Á.
Trong số 7 di sản văn hóa thế giới: có 4/5 di sản đáp ứng tiêu chuẩn II; 3/5 di sản đáp ứng tiêu chuẩn IV; 2/5 di sản đáp ứng tiêu chuẩn V; 3/5 di sản đáp ứng tiêu chuẩn III, 1 di sản đáp ứng tiêu chuẩn VI và chưa có di sản nào đáp ứng tiêu chuẩn I.
Cả ba di sản liên quan đến thiên nhiên đều đáp ứng tiêu chuẩn VIII còn các tiêu chuẩn VII, IX, X chỉ có ở một trong 3 di sản trên.
Di sản Vườn quốc gia Phong Nha - Kẻ Bàng là di sản thiên nhiên liên biên giới đầu tiên của Việt Nam.

[3] Từ Wikipedia - Việt Nam:
Việt Nam có diện tích 331.212 km², đường biên giới trên đất liền dài 4.639 km, đường bờ biển trải dài 3.260 km, có chung đường biên giới trên biển với Thái Lan qua vịnh Thái Lan và với Trung Quốc, Philippines, Indonesia, Brunei, Malaysia qua Biển Đông. Việt Nam tuyên bố chủ quyền đối với hai thực thể địa lí tranh chấp trên Biển Đông là các quần đảo Hoàng Sa (bị mất kiểm soát trên thực tế) và Trường Sa (kiểm soát một phần).
Khoảng cách giữa cực Bắc và cực Nam của Việt Nam theo đường chim bay là 1.650 km. Nơi có chiều ngang hẹp nhất ở Quảng Bình với chưa đầy 50 km. Đường biên giới đất liền dài hơn 4.600 km, trong đó, biên giới với Lào dài nhất (gần 2.100 km), tiếp đến là Trung Quốc và Campuchia. Tổng diện tích là 331.212 km² gồm toàn bộ phần đất liền và hải đảo cùng hơn 4.000 hòn đảo, bãi đá ngầm và cả hai quần đảo trên Biển Đông là Trường Sa (thuộc tỉnh Khánh Hòa) và Hoàng Sa (thuộc thành phố Đà Nẵng) mà nhà nước tuyên bố chủ quyền.
Địa hình Việt Nam có núi rừng chiếm khoảng 40%, đồi 40% và độ che phủ khoảng 75% diện tích đất nước. Có các dãy núi và cao nguyên như dãy Hoàng Liên Sơn, cao nguyên Sơn La ở phía bắc, dãy Bạch Mã và các cao nguyên theo dãy Trường Sơn ở phía nam. Mạng lưới sông, hồ ở vùng đồng bằng châu thổ hoặc miền núi phía Bắc và Tây Nguyên. Đồng bằng chiếm khoảng 1/4 diện tích, gồm các đồng bằng châu thổ như đồng bằng sông Hồng, sông Cửu Long và các vùng đồng bằng ven biển miền Trung, là vùng tập trung dân cư. Đất canh tác chiếm 17% tổng diện tích đất Việt Nam.
Đất chủ yếu là đất ferralit vùng đồi núi (ở Tây Nguyên hình thành trên đá bazan) và đất phù sa đồng bằng. Ven biển đồng bằng sông Hồng và sông Cửu Long tập trung đất phèn. Rừng ở Việt Nam chủ yếu là rừng rậm nhiệt đới khu vực đồi núi còn vùng đất thấp ven biển có rừng ngập mặn. Đất liền có các mỏ khoáng sản như phosphat, vàng. Than đá có nhiều nhất ở Quảng Ninh. Sắt ở Thái Nguyên, Hà Tĩnh. Ở biển có các mỏ dầu và khí tự nhiên.

Số lượng khách du lịch đến Việt Nam tăng nhanh nhất trong vòng 10 năm từ 2000–2010. Năm 2013, có gần 7,6 triệu lượt khách quốc tế đến Việt Nam và năm 2017, có hơn 10 triệu lượt khách quốc tế đến Việt Nam, các thị trường lớn nhất là Trung Quốc, Hàn Quốc, Nhật Bản, Hoa Kỳ và Đài Loan.
Việt Nam có các điểm du lịch từ Bắc đến Nam, từ miền núi tới đồng bằng, từ các thắng cảnh thiên nhiên tới các di tích văn hóa lịch sử. Các điểm du lịch miền núi như Sa Pa, Bà Nà, Đà Lạt. Các điểm du lịch ở các bãi biển như Đà Nẵng, Nha Trang, Vũng Tàu và các đảo như Cát Bà, Côn Đảo, Lý Sơn.
```


## Fine-tuning VLM cho Answer Generation

Theo framework từ [bài viết gốc](https://towardsdatascience.com/a-simple-framework-for-rag-enhanced-visual-question-answering-06768094762e/), có thể fine-tune VLM để cải thiện khả năng answer generation.

### Setup Fine-tuning

1. **Convert dữ liệu VQA**:
```bash
cd code
python scripts/convert_vqa_to_llamafactory.py \
    --input ../data/vqa.json \
    --output ../data/vqa_llamafactory.json \
    --split --val-ratio 0.1
```

2. **Fine-tune trên Kaggle**:
   - Sử dụng notebook `notebooks/4_FineTune_VQA.ipynb`
   - Config: `finetuning/llamafactory_config.yaml`
   - **Frozen vision encoder**: Chỉ train language part
   - **4-bit quantization**: Phù hợp cho T4 GPU

Xem chi tiết tại:
- [README_FINETUNING.md](code/finetuning/README_FINETUNING.md) - Hướng dẫn đầy đủ
- [QUICKSTART_FINETUNING.md](code/finetuning/QUICKSTART_FINETUNING.md) - Quick start Tài liệu tham khảo

- [Qwen2VL Model Card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [RAG Framework for VQA](https://towardsdatascience.com/a-simple-framework-for-rag-enhanced-visual-question-answering-06768094762e/)
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) - Framework cho fine-tuning
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)


## 👥 Tác giả

NLP Final Project - RAG-Enhanced VQA for Vietnamese History & Culture

## 📄 License

MIT License

