"""
Retrieval Module: Optimized RAG with Pre-built FAISS Index & Wikipedia Fallback
"""
import json
import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# Import cấu hình
from .config import (
    KB_JSON_PATH, RETRIEVAL_METHOD, TOP_K_RETRIEVE,
    BM25_K1, BM25_B, WIKIPEDIA_LANG, WIKIPEDIA_FALLBACK,
    VIETNAMESE_EMBEDDING_MODEL, DEVICE
)

# Đường dẫn mặc định tới Vector DB (folder chứa .index và _metadata.pkl)
# Bạn hãy đảm bảo folder này tồn tại và chứa file tải từ Kaggle về
VECTOR_DB_PATH = Path("models/vector_db") 

logger = logging.getLogger(__name__)

class RetrievalModule:
    """Module for knowledge retrieval using Pre-built FAISS, BM25 and Wikipedia"""
    
    def __init__(
        self,
        kb_path: Path = KB_JSON_PATH,
        vector_db_path: Path = VECTOR_DB_PATH,
        method: str = RETRIEVAL_METHOD,
        top_k: int = TOP_K_RETRIEVE,
        vision_module: Optional[object] = None
    ):
        self.method = method
        self.top_k = top_k
        self.kb_path = kb_path
        self.vector_db_path = vector_db_path
        self.vision_module = vision_module
        
        # 1. Load Metadata (Load file pickle sạch thay vì json gốc)
        self.kb_data = self._load_metadata()
        
        # 2. Init Retrieval Engines
        self.bm25 = None
        self.embedding_model = None
        self.index = None # FAISS index
        
        if method in ["bm25", "hybrid"]:
            self._init_bm25()
            
        if method in ["embedding", "hybrid"]:
            self._init_faiss_embedding()
            
        # 3. Init Wikipedia
        if WIKIPEDIA_FALLBACK:
            self._init_wikipedia()
        else:
            self.wikipedia = None

    def _load_metadata(self) -> List[Dict]:
        """Load metadata from pickle if available, else load raw JSON"""
        # Giả định file metadata tên là vector_db_metadata.pkl (khớp với code build trên Kaggle)
        # Nếu file bạn tải về tên khác, hãy sửa lại dòng này
        meta_path = str(self.vector_db_path) + "_metadata.pkl"
        
        # Fallback check tên file nếu không tìm thấy (do lúc download zip có thể đổi tên)
        if not os.path.exists(meta_path):
             # Thử tìm file .pkl bất kỳ trong thư mục vector_db
             pkl_files = list(Path(self.vector_db_path).parent.glob("*.pkl")) + list(Path(self.vector_db_path).glob("*.pkl"))
             if pkl_files:
                 meta_path = str(pkl_files[0])

        try:
            if os.path.exists(meta_path):
                logger.info(f"Loading metadata from {meta_path}...")
                with open(meta_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Loaded {len(data)} entries from metadata.")
                return data
            else:
                logger.warning(f"Metadata not found at {meta_path}. Falling back to raw JSON at {self.kb_path}.")
                with open(self.kb_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return []

    def _init_bm25(self):
        """Initialize BM25 from loaded metadata"""
        try:
            from rank_bm25 import BM25Okapi
            from underthesea import word_tokenize
            
            if not self.kb_data: return

            logger.info("Building BM25 index in memory...")
            tokenized_docs = []
            for item in self.kb_data:
                text = f"{item.get('entity', '')} {item.get('facts', '')} {item.get('summary', '')}"
                tokenized_docs.append(word_tokenize(text.lower()))
            
            if tokenized_docs:
                self.bm25 = BM25Okapi(tokenized_docs, k1=BM25_K1, b=BM25_B)
                logger.info("BM25 initialized successfully.")
            else:
                logger.warning("No documents to build BM25.")
                
        except Exception as e:
            logger.error(f"Error initializing BM25: {e}")

    def _init_faiss_embedding(self):
        """Load Pre-built FAISS Index and Embedding Model"""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            
            # Load Model (chỉ để encode query)
            logger.info(f"Loading embedding model: {VIETNAMESE_EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(VIETNAMESE_EMBEDDING_MODEL)
            
            # Load Index
            index_file = str(self.vector_db_path) + ".index"
            
            # Fallback check tên file
            if not os.path.exists(index_file):
                 idx_files = list(Path(self.vector_db_path).parent.glob("*.index")) + list(Path(self.vector_db_path).glob("*.index"))
                 if idx_files:
                     index_file = str(idx_files[0])

            if os.path.exists(index_file):
                logger.info(f"Loading FAISS index from {index_file}...")
                self.index = faiss.read_index(index_file)
                logger.info(f"FAISS index loaded. Total vectors: {self.index.ntotal}")
            else:
                logger.error(f"FAISS index file not found at {index_file}. Please run 1_Build_Index.ipynb first!")
                self.index = None
                
        except Exception as e:
            logger.error(f"Error initializing FAISS: {e}")

    def _init_wikipedia(self):
        try:
            import wikipedia
            wikipedia.set_lang(WIKIPEDIA_LANG)
            self.wikipedia = wikipedia
            logger.info(f"Wikipedia initialized for language: {WIKIPEDIA_LANG}")
        except ImportError:
            logger.warning("wikipedia package not available")
            self.wikipedia = None
        except Exception as e:
            logger.error(f"Error initializing Wikipedia: {e}")
            self.wikipedia = None

    def _bm25_search(self, query: str) -> List[Tuple[int, float]]:
        if self.bm25 is None: return []
        try:
            from underthesea import word_tokenize
            tokens = word_tokenize(query.lower())
            scores = self.bm25.get_scores(tokens)
            top_indices = np.argsort(scores)[::-1][:self.top_k]
            return [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        except Exception as e:
            logger.error(f"BM25 Search Error: {e}")
            return []

    def _embedding_search(self, query: str) -> List[Tuple[int, float]]:
        """Search using FAISS"""
        if self.embedding_model is None or self.index is None:
            return []
        
        try:
            # Encode Query
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            import faiss
            faiss.normalize_L2(query_embedding) # Normalize query logic
            
            # Search
            D, I = self.index.search(query_embedding, self.top_k)
            
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx != -1: 
                    results.append((int(idx), float(score)))
            return results
        except Exception as e:
            logger.error(f"FAISS Search Error: {e}")
            return []

    # --- Các hàm hỗ trợ Wikipedia & VLM (Giữ nguyên từ code cũ) ---
    
    def _generate_keywords_with_vlm(self, image, question: str, caption: str = "", ocr_text: str = "") -> str:
        if self.vision_module is None:
            return f"{question} {caption} {ocr_text}".strip()
        
        try:
            prompt = (
                f"Dựa vào câu hỏi: '{question}'\n"
                f"Mô tả hình ảnh: {caption}\n"
                f"Văn bản trong ảnh: {ocr_text}\n\n"
                "Hãy tạo ra 3-5 từ khóa quan trọng nhất (tên riêng, địa danh, sự kiện lịch sử) "
                "để tìm kiếm thông tin trên Wikipedia. "
                "Chỉ trả về các từ khóa, cách nhau bằng dấu phẩy."
            )
            
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text = self.vision_module.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            images = [image]
            inputs = self.vision_module.processor(text=[text], images=images, padding=True, return_tensors="pt")
            
            import torch
            # Device handling logic
            device = DEVICE
            if hasattr(self.vision_module.model, 'device'): device = self.vision_module.model.device
            
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.vision_module.model.generate(**inputs, max_new_tokens=64, do_sample=False)
            
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
            keywords = self.vision_module.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
            
            logger.info(f"VLM Generated keywords: {keywords}")
            return keywords
            
        except Exception as e:
            logger.error(f"Error generating keywords: {e}")
            return f"{question} {caption} {ocr_text}".strip()

    def _chunk_wikipedia_content(self, content: str, chunk_size: int = 500) -> List[str]:
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para: continue
            if current_chunk and len(current_chunk) + len(para) > chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk = (current_chunk + "\n\n" + para) if current_chunk else para
        if current_chunk: chunks.append(current_chunk.strip())
        return chunks

    def _rank_chunks_by_similarity(self, chunks: List[str], query: str, keywords: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if not chunks or self.embedding_model is None:
            return [(chunk, 1.0) for chunk in chunks[:top_k]]
        try:
            search_text = f"{query} {keywords}".strip()
            search_embedding = self.embedding_model.encode(search_text, convert_to_numpy=True)
            chunk_embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)
            
            scores = np.dot(chunk_embeddings, search_embedding) / (
                np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(search_embedding)
            )
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(chunks[idx], float(scores[idx])) for idx in top_indices]
        except:
            return [(chunk, 1.0) for chunk in chunks[:top_k]]

    def _wikipedia_search(self, query: str, image=None, caption: str = "", ocr_text: str = "", max_results: int = 3) -> List[Dict]:
        if self.wikipedia is None: return []
        try:
            if image is not None and self.vision_module is not None:
                search_keywords = self._generate_keywords_with_vlm(image, query, caption, ocr_text)
            else:
                search_keywords = f"{query} {caption} {ocr_text}".strip()
            
            search_results = self.wikipedia.search(search_keywords, results=max_results)
            retrieved = []
            
            for page_title in search_results:
                try:
                    page = self.wikipedia.page(page_title)
                    chunks = self._chunk_wikipedia_content(page.content)
                    ranked = self._rank_chunks_by_similarity(chunks, query, search_keywords, top_k=2)
                    best_content = "\n\n".join([chunk for chunk, _ in ranked]) or page.content[:500]
                    
                    retrieved.append({
                        "title": page_title, "content": best_content,
                        "url": page.url, "source": "wikipedia", "keywords_used": search_keywords
                    })
                except: continue
            return retrieved
        except Exception as e:
            logger.error(f"Wiki Error: {e}")
            return []

    def retrieve(self, query: str, caption: str = "", ocr_text: str = "", image=None) -> List[Dict]:
        """Main retrieval function"""
        search_query = f"{query} {caption} {ocr_text}".strip()
        
        results = []
        
        # --- Logic BM25 / Embedding / Hybrid ---
        if self.method == "bm25":
            hits = self._bm25_search(search_query)
            for idx, score in hits:
                if idx < len(self.kb_data):
                    item = self.kb_data[idx].copy()
                    item.update({"score": score, "source": "kb_bm25"})
                    results.append(item)
                
        elif self.method == "embedding":
            hits = self._embedding_search(search_query)
            for idx, score in hits:
                if idx < len(self.kb_data):
                    item = self.kb_data[idx].copy()
                    item.update({"score": score, "source": "kb_embedding"})
                    results.append(item)
                
        elif self.method == "hybrid":
            bm25_hits = dict(self._bm25_search(search_query))
            emb_hits = dict(self._embedding_search(search_query))
            
            all_indices = set(bm25_hits.keys()) | set(emb_hits.keys())
            hybrid_scores = []
            
            for idx in all_indices:
                s1 = bm25_hits.get(idx, 0.0)
                s2 = emb_hits.get(idx, 0.0)
                final_score = s1 * 0.3 + s2 * 0.7 * 10
                hybrid_scores.append((idx, final_score))
            
            hybrid_scores.sort(key=lambda x: x[1], reverse=True)
            for idx, score in hybrid_scores[:self.top_k]:
                if idx < len(self.kb_data):
                    item = self.kb_data[idx].copy()
                    item.update({"score": score, "source": "kb_hybrid"})
                    results.append(item)

        # --- Wikipedia Fallback ---
        if WIKIPEDIA_FALLBACK and len(results) < self.top_k:
            wiki_results = self._wikipedia_search(
                query=query, image=image, caption=caption, 
                ocr_text=ocr_text, max_results=self.top_k - len(results)
            )
            results.extend(wiki_results)
        
        return results[:self.top_k]