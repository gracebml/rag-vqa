"""
Retrieval Module: RAG with BM25, Embedding Search, and Wikipedia
Enhanced with VLM-based keyword generation for Wikipedia search
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np
import re

from .config import (
    KB_JSON_PATH, RETRIEVAL_METHOD, TOP_K_RETRIEVE,
    BM25_K1, BM25_B, WIKIPEDIA_LANG, WIKIPEDIA_FALLBACK,
    VIETNAMESE_EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)


class RetrievalModule:
    """Module for knowledge retrieval using BM25, Embedding, and Wikipedia"""
    
    def __init__(
        self,
        kb_path: Path = KB_JSON_PATH,
        method: str = RETRIEVAL_METHOD,
        top_k: int = TOP_K_RETRIEVE,
        vision_module: Optional[object] = None
    ):
        """
        Initialize retrieval module
        
        Args:
            kb_path: Path to knowledge base JSON file
            method: Retrieval method ("bm25", "embedding", "hybrid")
            top_k: Number of top results to retrieve
            vision_module: Optional VisionModule instance for VLM-based keyword generation
        """
        self.method = method
        self.top_k = top_k
        self.kb_path = kb_path
        self.vision_module = vision_module  # VLM for keyword generation
        
        # Load knowledge base
        self.kb_data = self._load_knowledge_base()
        
        # Initialize retrieval components
        self.bm25 = None
        self.embedding_model = None
        self.embeddings = None
        
        if method in ["bm25", "hybrid"]:
            self._init_bm25()
        
        if method in ["embedding", "hybrid"]:
            self._init_embedding()
        
        # Initialize Wikipedia
        if WIKIPEDIA_FALLBACK:
            self._init_wikipedia()
        else:
            self.wikipedia = None
    
    def _load_knowledge_base(self) -> List[Dict]:
        """Load knowledge base from JSON file"""
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} entries from knowledge base")
            return data
        except FileNotFoundError:
            logger.warning(f"Knowledge base not found at {self.kb_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return []
    
    def _init_bm25(self):
        """Initialize BM25 retriever"""
        try:
            from rank_bm25 import BM25Okapi
            from underthesea import word_tokenize
            
            # Tokenize all documents
            self.tokenized_docs = []
            for item in self.kb_data:
                # Combine entity, facts, summary into searchable text
                text = f"{item.get('entity', '')} {item.get('facts', '')} {item.get('summary', '')}"
                tokens = word_tokenize(text.lower())
                self.tokenized_docs.append(tokens)
            
            if self.tokenized_docs:
                self.bm25 = BM25Okapi(
                    self.tokenized_docs,
                    k1=BM25_K1,
                    b=BM25_B
                )
                logger.info("BM25 initialized")
            else:
                logger.warning("No documents for BM25")
                
        except ImportError:
            logger.warning("rank_bm25 or underthesea not available")
        except Exception as e:
            logger.error(f"Error initializing BM25: {e}")
    
    def _init_embedding(self):
        """Initialize embedding model and create embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {VIETNAMESE_EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(VIETNAMESE_EMBEDDING_MODEL)
            
            # Create embeddings for all documents
            if self.kb_data:
                texts = []
                for item in self.kb_data:
                    text = f"{item.get('entity', '')} {item.get('facts', '')} {item.get('summary', '')}"
                    texts.append(text)
                
                logger.info("Creating embeddings for knowledge base...")
                self.embeddings = self.embedding_model.encode(
                    texts,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
                logger.info("Embeddings created successfully")
            else:
                self.embeddings = None
                
        except ImportError:
            logger.warning("sentence-transformers not available")
        except Exception as e:
            logger.error(f"Error initializing embedding: {e}")
    
    def _init_wikipedia(self):
        """Initialize Wikipedia search"""
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
        """Search using BM25"""
        if self.bm25 is None:
            return []
        
        try:
            from underthesea import word_tokenize
            query_tokens = word_tokenize(query.lower())
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:self.top_k]
            results = [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]
            return results
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def _embedding_search(self, query: str) -> List[Tuple[int, float]]:
        """Search using embeddings"""
        if self.embedding_model is None or self.embeddings is None:
            return []
        
        try:
            query_embedding = self.embedding_model.encode(
                query,
                convert_to_numpy=True
            )
            
            # Compute cosine similarity
            scores = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:self.top_k]
            results = [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]
            return results
        except Exception as e:
            logger.error(f"Error in embedding search: {e}")
            return []
    
    def _generate_keywords_with_vlm(
        self,
        image,
        question: str,
        caption: str = "",
        ocr_text: str = ""
    ) -> str:
        """
        Generate search keywords using VLM to capture meaning of question and image.
        Based on vision_rag approach: https://github.com/GabrieleSgroi/vision_rag
        
        Args:
            image: PIL Image object
            question: User question
            caption: Image caption
            ocr_text: OCR text from image
            
        Returns:
            Generated keywords string for Wikipedia search
        """
        if self.vision_module is None:
            # Fallback: combine question, caption, and OCR
            keywords = f"{question} {caption} {ocr_text}".strip()
            logger.warning("Vision module not available, using fallback keyword generation")
            return keywords
        
        try:
            # Use VLM to generate keywords capturing the meaning
            prompt = (
                f"Dựa vào câu hỏi: '{question}'\n"
                f"Mô tả hình ảnh: {caption}\n"
                f"Văn bản trong ảnh: {ocr_text}\n\n"
                "Hãy tạo ra 3-5 từ khóa quan trọng nhất (tên riêng, địa danh, sự kiện lịch sử, "
                "công trình văn hóa) để tìm kiếm thông tin trên Wikipedia. "
                "Chỉ trả về các từ khóa, cách nhau bằng dấu phẩy, không giải thích thêm."
            )
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Use vision module's processor and model
            text = self.vision_module.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Extract images from messages
            images = []
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for content in msg["content"]:
                        if content.get("type") == "image":
                            images.append(content["image"])
            
            inputs = self.vision_module.processor(
                text=[text],
                images=images if images else None,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            import torch
            from .config import DEVICE
            # Handle device for models with device_map="auto"
            if hasattr(self.vision_module.model, 'device'):
                device = self.vision_module.model.device
            elif hasattr(self.vision_module.model, 'hf_device_map'):
                # Model is split across devices, use first device or default
                device = DEVICE
            else:
                device = DEVICE
            
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate keywords
            with torch.no_grad():
                generated_ids = self.vision_module.model.generate(
                    **inputs,
                    max_new_tokens=64,  # Short keywords
                    do_sample=False
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            keywords = self.vision_module.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            logger.info(f"Generated keywords: {keywords}")
            return keywords
            
        except Exception as e:
            logger.error(f"Error generating keywords with VLM: {e}")
            # Fallback to simple combination
            keywords = f"{question} {caption} {ocr_text}".strip()
            return keywords
    
    def _chunk_wikipedia_content(self, content: str, chunk_size: int = 500) -> List[str]:
        """
        Split Wikipedia content into chunks for better retrieval.
        
        Args:
            content: Full Wikipedia page content
            chunk_size: Target size for each chunk in characters
            
        Returns:
            List of content chunks
        """
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed chunk size, save current chunk
            if current_chunk and len(current_chunk) + len(para) > chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _rank_chunks_by_similarity(
        self,
        chunks: List[str],
        query: str,
        keywords: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Rank chunks by semantic similarity to query and keywords.
        
        Args:
            chunks: List of content chunks
            query: Original question
            keywords: Generated keywords
            top_k: Number of top chunks to return
            
        Returns:
            List of (chunk, score) tuples sorted by score
        """
        if not chunks or self.embedding_model is None:
            # Fallback: return first chunks
            return [(chunk, 1.0) for chunk in chunks[:top_k]]
        
        try:
            # Combine query and keywords for search
            search_text = f"{query} {keywords}".strip()
            
            # Encode search text
            search_embedding = self.embedding_model.encode(
                search_text,
                convert_to_numpy=True
            )
            
            # Encode all chunks
            chunk_embeddings = self.embedding_model.encode(
                chunks,
                convert_to_numpy=True
            )
            
            # Compute cosine similarity
            scores = np.dot(chunk_embeddings, search_embedding) / (
                np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(search_embedding)
            )
            
            # Get top-k chunks
            top_indices = np.argsort(scores)[::-1][:top_k]
            ranked_chunks = [(chunks[idx], float(scores[idx])) for idx in top_indices]
            
            return ranked_chunks
            
        except Exception as e:
            logger.error(f"Error ranking chunks: {e}")
            # Fallback: return first chunks
            return [(chunk, 1.0) for chunk in chunks[:top_k]]
    
    def _wikipedia_search(
        self,
        query: str,
        image=None,
        caption: str = "",
        ocr_text: str = "",
        max_results: int = 3
    ) -> List[Dict]:
        """
        Search Wikipedia for additional information with VLM-generated keywords.
        Enhanced with chunking and semantic ranking.
        
        Args:
            query: Original question
            image: PIL Image object (optional, for VLM keyword generation)
            caption: Image caption
            ocr_text: OCR text from image
            max_results: Maximum number of Wikipedia pages to retrieve
            
        Returns:
            List of retrieved Wikipedia documents
        """
        if self.wikipedia is None:
            return []
        
        try:
            # Step 1: Generate keywords using VLM if available
            if image is not None and self.vision_module is not None:
                search_keywords = self._generate_keywords_with_vlm(
                    image=image,
                    question=query,
                    caption=caption,
                    ocr_text=ocr_text
                )
                logger.info(f"Using VLM-generated keywords: {search_keywords}")
            else:
                # Fallback: use query + caption + OCR
                search_keywords = f"{query} {caption} {ocr_text}".strip()
                logger.info(f"Using fallback keywords: {search_keywords}")
            
            # Step 2: Search Wikipedia with keywords
            search_results = self.wikipedia.search(search_keywords, results=max_results)
            
            retrieved = []
            for page_title in search_results:
                try:
                    page = self.wikipedia.page(page_title)
                    full_content = page.content
                    
                    # Step 3: Split content into chunks
                    chunks = self._chunk_wikipedia_content(full_content, chunk_size=500)
                    
                    # Step 4: Rank chunks by semantic similarity
                    ranked_chunks = self._rank_chunks_by_similarity(
                        chunks=chunks,
                        query=query,
                        keywords=search_keywords,
                        top_k=2  # Get top 2 chunks per page
                    )
                    
                    # Combine top chunks
                    best_content = "\n\n".join([chunk for chunk, _ in ranked_chunks])
                    
                    # If no chunks ranked, use first 500 chars as fallback
                    if not best_content:
                        best_content = full_content[:500]
                    
                    retrieved.append({
                        "title": page_title,
                        "content": best_content,
                        "url": page.url,
                        "source": "wikipedia",
                        "keywords_used": search_keywords
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing Wikipedia page {page_title}: {e}")
                    continue
            
            return retrieved
            
        except Exception as e:
            logger.error(f"Error in Wikipedia search: {e}")
            return []
    
    def retrieve(
        self,
        query: str,
        caption: str = "",
        ocr_text: str = "",
        image=None
    ) -> List[Dict]:
        """
        Retrieve relevant knowledge base entries
        
        Args:
            query: User question
            caption: Image caption
            ocr_text: OCR text from image
            image: PIL Image object (optional, for VLM keyword generation in Wikipedia search)
            
        Returns:
            List of retrieved documents with metadata
        """
        # Combine query components
        search_query = f"{query} {caption} {ocr_text}".strip()
        
        results = []
        
        if self.method == "bm25":
            bm25_results = self._bm25_search(search_query)
            for idx, score in bm25_results:
                if idx < len(self.kb_data):
                    item = self.kb_data[idx].copy()
                    item["score"] = score
                    item["source"] = "kb_bm25"
                    results.append(item)
        
        elif self.method == "embedding":
            emb_results = self._embedding_search(search_query)
            for idx, score in emb_results:
                if idx < len(self.kb_data):
                    item = self.kb_data[idx].copy()
                    item["score"] = score
                    item["source"] = "kb_embedding"
                    results.append(item)
        
        elif self.method == "hybrid":
            # Combine BM25 and embedding results
            bm25_results = self._bm25_search(search_query)
            emb_results = self._embedding_search(search_query)
            
            # Merge and deduplicate
            combined = {}
            for idx, score in bm25_results:
                if idx not in combined:
                    combined[idx] = {"bm25_score": score, "emb_score": 0.0}
                else:
                    combined[idx]["bm25_score"] = score
            
            for idx, score in emb_results:
                if idx not in combined:
                    combined[idx] = {"bm25_score": 0.0, "emb_score": score}
                else:
                    combined[idx]["emb_score"] = score
            
            # Normalize and combine scores
            for idx, scores in combined.items():
                # Simple average (can be improved with weighted combination)
                combined_score = (scores["bm25_score"] + scores["emb_score"]) / 2
                if idx < len(self.kb_data):
                    item = self.kb_data[idx].copy()
                    item["score"] = combined_score
                    item["source"] = "kb_hybrid"
                    results.append((idx, combined_score, item))
            
            # Sort by combined score
            results.sort(key=lambda x: x[1], reverse=True)
            results = [item for _, _, item in results[:self.top_k]]
        
        # Add Wikipedia results if enabled and local KB has few results
        # Enhanced with VLM keyword generation
        if WIKIPEDIA_FALLBACK and len(results) < self.top_k:
            wiki_results = self._wikipedia_search(
                query=query,
                image=image,
                caption=caption,
                ocr_text=ocr_text,
                max_results=self.top_k - len(results)
            )
            results.extend(wiki_results)
        
        return results[:self.top_k]

