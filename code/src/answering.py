"""
Answering Module: Generate answers using Qwen2VL with 4-bit quantization
"""
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from typing import List, Dict, Optional
import logging

from .config import (
    QWEN2VL_MODEL_NAME, QWEN2VL_4BIT, DEVICE,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P
)

logger = logging.getLogger(__name__)


class AnsweringModule:
    """Module for generating answers using Qwen2VL"""
    
    def __init__(
        self,
        model_name: str = QWEN2VL_MODEL_NAME,
        use_4bit: bool = QWEN2VL_4BIT
    ):
        """
        Initialize answering module with Qwen2VL
        
        Args:
            model_name: HuggingFace model name
            use_4bit: Whether to use 4-bit quantization (for T4 GPU)
        """
        logger.info(f"Loading Qwen2VL model: {model_name}")
        logger.info(f"4-bit quantization: {use_4bit}")
        
        # Load processor
        self.processor = Qwen2VLProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Configure quantization if needed
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto"
        }
        
        if use_4bit:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        self.model.eval()
        logger.info("Answering module loaded successfully")
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "Không có thông tin liên quan."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.get("source", "unknown")
            
            if source == "wikipedia":
                context_parts.append(
                    f"[{i}] Từ Wikipedia - {doc.get('title', '')}:\n{doc.get('content', '')}"
                )
            else:
                entity = doc.get("entity", "")
                facts = doc.get("facts", "")
                summary = doc.get("summary", "")
                
                context_text = f"[{i}] {entity}"
                if summary:
                    context_text += f": {summary}"
                if facts:
                    context_text += f"\n{facts}"
                
                context_parts.append(context_text)
        
        return "\n\n".join(context_parts)
    
    def generate_answer(
        self,
        image: Image.Image,
        question: str,
        caption: str,
        ocr_text: str,
        context: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P
    ) -> str:
        """
        Generate answer using Qwen2VL with RAG context
        
        Args:
            image: PIL Image object
            question: User question
            caption: Image caption
            ocr_text: OCR text from image
            context: Retrieved knowledge context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated answer string
        """
        # Build prompt following the framework from the article
        prompt = (
            f"Dựa vào mô tả hình ảnh: {caption}\n\n"
            f"Thông tin văn bản trong ảnh: {ocr_text}\n\n"
            f"Kiến thức lịch sử và văn hóa:\n{context}\n\n"
            f"Hãy trả lời câu hỏi sau bằng tiếng Việt, rõ ràng và có cấu trúc: {question}\n\n"
            "Nếu phù hợp, hãy cung cấp giải thích chi tiết, ý nghĩa văn hóa và bối cảnh lịch sử."
        )
        
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý VQA chuyên về lịch sử và văn hóa Việt Nam. "
                    "Hãy trả lời câu hỏi dựa trên hình ảnh, thông tin văn bản, và kiến thức được cung cấp. "
                    "Trả lời bằng tiếng Việt, rõ ràng và có cấu trúc."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]
        
        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Extract images from messages
            images = []
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for content in msg["content"]:
                        if content.get("type") == "image":
                            images.append(content["image"])
            
            inputs = self.processor(
                text=[text],
                images=images if images else None,
                padding=True,
                return_tensors="pt"
            )
            # Move to device
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Lỗi khi tạo câu trả lời: {str(e)}"

