"""
Vision Module: Image Captioning and OCR using Qwen2VL
"""
import torch
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from typing import List, Optional, Dict
import logging

from .config import QWEN2VL_MODEL_NAME, DEVICE

logger = logging.getLogger(__name__)

class VisionModule:
    """Module for vision-to-text conversion: Captioning and OCR using a unified Qwen2VL model"""
    
    def __init__(self, model_name: str = QWEN2VL_MODEL_NAME, use_4bit: bool = True):
        """
        Initialize vision module with Qwen2VL for both Captioning and OCR
        """
        logger.info(f"Loading Qwen2VL model: {model_name}")
        logger.info(f"4-bit quantization: {use_4bit}")
        
        # Configure quantization
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "attn_implementation": "sdpa"
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
        
        # Load processor
        min_pixels = 256 * 28 * 28
        max_pixels = 640 * 28 * 28
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True
        )
        
        logger.info("Vision module loaded successfully (Unified Qwen2VL)")

    def _generate_response(self, image: Image.Image, prompt_text: str, max_new_tokens: int = 512) -> str:
        """Helper method to handle Qwen2VL generation logic"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        
        try:
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs = [msg["content"][0]["image"] for msg in messages]
            
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
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
            logger.error(f"Error during Qwen2VL generation: {e}")
            return ""

    def generate_caption(self, image: Image.Image) -> str:
        """Generate Vietnamese caption"""
        prompt = (
            "Hãy mô tả chi tiết hình ảnh này bằng tiếng Việt. "
            "Bao gồm các đối tượng, hành động, bối cảnh. "
            "Nếu là địa danh nổi tiếng ở Việt Nam, hãy nêu tên cụ thể."
        )
        return self._generate_response(image, prompt, max_new_tokens=256)

    def extract_ocr(self, image: Image.Image) -> str:
        """
        Extract and clean text from image using Qwen2VL.
        Combines extraction and correction in one pass.
        """
        prompt = (
            "Đọc và trích xuất toàn bộ văn bản tiếng Việt xuất hiện trong hình ảnh này. "
            "Yêu cầu:\n"
            "1. Chỉ xuất ra nội dung văn bản tìm thấy.\n"
            "2. Tự động sửa các lỗi chính tả hoặc thay những ký tự lạ bằng những ký tự Tiếng Việt tương tự\n"
            "3. Nếu không có văn bản hoặc văn bản không có ý nghĩa, hãy trả về 'NO_TEXT'.\n"
            "4. Không thêm lời dẫn, chỉ trả về văn bản."
        )
        
        logger.info("Running Unified OCR extraction...")
        result = self._generate_response(image, prompt, max_new_tokens=512)
        
        if "NO_TEXT" in result or not result.strip():
            logger.info("No meaningful text found.")
            return ""
            
        logger.info(f"OCR Result: {result[:100]}...")
        return result

    def process_image(self, image: Image.Image) -> Dict[str, str]:
        """Process image to get both caption and OCR text"""
        logger.info("Processing image...")
        
        caption = self.generate_caption(image)
        ocr_text = self.extract_ocr(image)
        
        return {
            "caption": caption,
            "ocr": ocr_text
        }