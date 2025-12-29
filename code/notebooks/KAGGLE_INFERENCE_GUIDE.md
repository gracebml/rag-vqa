# Hướng dẫn chạy Inference trên Kaggle

Hướng dẫn chi tiết để push code và chạy inference trên Kaggle.

##  Chuẩn bị

### 1. Chuẩn bị dữ liệu cần upload lên Kaggle

Bạn cần tạo các Kaggle Datasets sau:

#### a) Dataset chứa source code
- **Tên dataset**: `vqa-code` (hoặc tên khác, nhớ điều chỉnh trong notebook)
- **Cấu trúc**:
  ```
  vqa-code/
  └── code/
      └── src/
          ├── __init__.py
          ├── config.py
          ├── pipeline.py
          ├── vision.py
          ├── answering.py
          └── retrieval.py
  ```

#### b) Dataset chứa knowledge base
- **Tên dataset**: `vietnamese-knowledge-base`
- **File**: `knowledge_base.json`

#### c) Dataset chứa ảnh test
- **Tên dataset**: `vqa-images`
- **Cấu trúc**: 
  ```
  vqa-images/
  └── images_flat/
      ├── 004122.jpg
      ├── 004129.jpg
      └── ...
  ```

#### d) Dataset chứa vector database (nếu đã build)
- **Tên dataset**: `vqa-vector-db`
- **Files**:
  - `vector_db.index`
  - `vector_db_metadata.pkl`
  - `vector_db_config.json`

##  Các bước thực hiện

### Bước 1: Tạo Kaggle Datasets

1. **Upload source code**:
   - Vào [Kaggle Datasets](https://www.kaggle.com/datasets)
   - Click "New Dataset"
   - Upload thư mục `code/src` (hoặc toàn bộ `code`)
   - Đặt tên dataset: `vqa-code`
   - Publish dataset

2. **Upload knowledge base**:
   - Tạo dataset mới: `vietnamese-knowledge-base`
   - Upload file `knowledge_base.json`

3. **Upload ảnh**:
   - Tạo dataset mới: `vqa-images`
   - Upload thư mục `images_flat` hoặc zip file chứa ảnh

4. **Upload vector DB (nếu có)**:
   - Tạo dataset mới: `vqa-vector-db`
   - Upload các file index đã build từ notebook `1_Build_Index.ipynb`

### Bước 2: Tạo Notebook trên Kaggle

1. Vào [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Chọn GPU accelerator (T4 hoặc P100)
4. Enable Internet (để download model từ HuggingFace)

### Bước 3: Upload notebook và cấu hình

1. **Upload notebook**:
   - Upload file `3_Inference_Sample.ipynb` vào Kaggle
   - Hoặc copy nội dung vào notebook mới

2. **Add datasets**:
   - Click "Add Data" trong notebook
   - Tìm và thêm các datasets:
     - `vqa-code`
     - `vietnamese-knowledge-base`
     - `vqa-images`
     - `vqa-vector-db` (nếu có)

3. **Điều chỉnh đường dẫn**:
   - Mở cell "5. Cấu hình đường dẫn"
   - Điều chỉnh tên datasets nếu khác với mặc định:
     ```python
     KB_PATH = "/kaggle/input/vietnamese-knowledge-base/knowledge_base.json"
     IMAGES_DIR = "/kaggle/input/vqa-images/images_flat"
     # ... etc
     ```

### Bước 4: Chạy notebook

1. **Chạy từng cell**:
   - Cell 1: Cài đặt thư viện
   - Cell 2-3: Copy/setup source code
   - Cell 4: Import và kiểm tra
   - Cell 5: Cấu hình đường dẫn
   - Cell 6: Khởi tạo pipeline (mất vài phút)
   - Cell 7: Chạy inference
   - Cell 8-10: Xem kết quả

2. **Lưu ý**:
   - Lần đầu chạy sẽ download Qwen2VL-7B model (~14GB) - mất thời gian
   - Model sẽ được cache, lần sau chạy nhanh hơn
   - Đảm bảo có đủ disk space (tối thiểu 20GB)

## 🔧 Troubleshooting

### Lỗi: "Module not found" hoặc "Import error"

**Giải pháp**:
1. Kiểm tra dataset `vqa-code` đã được add chưa
2. Kiểm tra đường dẫn trong cell 2-3
3. Đảm bảo cấu trúc thư mục đúng: `/kaggle/input/vqa-code/code/src/`

### Lỗi: "Image not found"

**Giải pháp**:
1. Kiểm tra dataset `vqa-images` đã được add chưa
2. Kiểm tra tên ảnh trong `SAMPLE_IMAGES` có đúng không
3. Kiểm tra đường dẫn `IMAGES_DIR`

### Lỗi: "Out of Memory" (OOM)

**Giải pháp**:
1. Đảm bảo đang dùng GPU (T4 hoặc P100)
2. Kiểm tra `use_4bit=True` trong pipeline initialization
3. Giảm số lượng ảnh test trong `SAMPLE_IMAGES`
4. Restart kernel và chạy lại

### Lỗi: "Knowledge base not found"

**Giải pháp**:
1. Kiểm tra dataset `vietnamese-knowledge-base` đã được add
2. Kiểm tra đường dẫn `KB_PATH`
3. Đảm bảo file `knowledge_base.json` có trong dataset

### Model download quá chậm

**Giải pháp**:
1. Đảm bảo Internet đã được enable
2. Lần đầu download sẽ lâu, lần sau sẽ dùng cache
3. Có thể pre-download model và upload lên dataset riêng

## 📝 Tùy chỉnh

### Thay đổi ảnh test

Sửa trong cell "5. Cấu hình đường dẫn":
```python
SAMPLE_IMAGES = [
    "your_image_1.jpg",
    "your_image_2.jpg",
    # ...
]
```

### Thay đổi câu hỏi

Sửa trong cell "5. Cấu hình đường dẫn":
```python
SAMPLE_QUESTIONS = [
    "Câu hỏi của bạn 1",
    "Câu hỏi của bạn 2",
    # ...
]
```

### Test với một ảnh cụ thể

Sử dụng cell "10. Test với một ảnh và câu hỏi tùy chỉnh":
```python
test_image_name = "your_image.jpg"
test_question = "Câu hỏi của bạn"
```

##  Tips

1. **Tối ưu thời gian chạy**:
   - Build vector index trước (notebook `1_Build_Index.ipynb`)
   - Upload vector DB lên dataset để không phải build lại

2. **Debug**:
   - Chạy từng cell một để dễ debug
   - Kiểm tra output của mỗi cell trước khi chạy cell tiếp theo

3. **Lưu kết quả**:
   - Kết quả được lưu trong `/kaggle/working/results/`
   - Download về máy local sau khi chạy xong

4. **Share notebook**:
   - Có thể public notebook để người khác sử dụng
   - Nhớ public các datasets cần thiết

##  Output

Sau khi chạy xong, bạn sẽ có:
- `inference_results.json`: Kết quả chi tiết dạng JSON
- `inference_results.csv`: Kết quả dạng CSV để xem dễ dàng

Các file này nằm trong `/kaggle/working/results/` và có thể download về máy.

