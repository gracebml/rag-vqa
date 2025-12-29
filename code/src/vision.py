"""
Vision Module: Image Captioning and OCR using Qwen2VL
"""
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from typing import List, Optional, Dict
import logging

# Support both relative and absolute imports (for Kaggle notebook)
try:
    from .config import USE_PADDLE_OCR, PADDLE_OCR_LANG, DEVICE
except ImportError:
    from config import USE_PADDLE_OCR, PADDLE_OCR_LANG, DEVICE

logger = logging.getLogger(__name__)


class VisionModule:
    """Module for vision-to-text conversion: Captioning and OCR"""
    
    def __init__(self, model_name: str = None, use_4bit: bool = True):
        """
        Initialize vision module with Qwen2VL
        
        Args:
            model_name: HuggingFace model name for Qwen2VL (default from config)
            use_4bit: Whether to use 4-bit quantization (recommended for T4 GPU)
        """
        # Use default from config if None
        if model_name is None:
            try:
                try:
                    from .config import QWEN2VL_MODEL_NAME
                except ImportError:
                    from config import QWEN2VL_MODEL_NAME
                model_name = QWEN2VL_MODEL_NAME
            except (ImportError, NameError):
                # Fallback if config import fails
                model_name = "Qwen/Qwen2-VL-7B-Instruct"
                logger.warning(f"Could not import QWEN2VL_MODEL_NAME from config, using default: {model_name}")
        
        logger.info(f"Loading Qwen2VL model: {model_name}")
        logger.info(f"4-bit quantization: {use_4bit}")
        
        # Configure quantization if needed
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,              # Enable 4-bit quantization
                bnb_4bit_use_double_quant=True, # Double quantization to save more memory
                bnb_4bit_quant_type="nf4",      # NF4 data type optimized for LLM
                bnb_4bit_compute_dtype=torch.float16  # Compute in float16 for speed
            )
        
        # Load model with quantization if enabled
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "attn_implementation": "sdpa"  # Use PyTorch's efficient attention
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
        
        # Load processor with optimized pixel settings for memory efficiency
        # The default range for visual tokens is 4-16384. Setting min_pixels and max_pixels
        # to a token count range of 256-1280 balances speed and memory usage.
        min_pixels = 256 * 28 * 28
        max_pixels = 640 * 28 * 28
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True
        )
        
        logger.info("Vision module loaded successfully")
        if use_4bit:
            logger.info("Model loaded with 4-bit quantization (~6GB VRAM)")
        
        # Initialize OCR
        self.ocr_engine = None
        if USE_PADDLE_OCR:
            try:
                from paddleocr import PaddleOCR
                self.ocr_engine = PaddleOCR(use_angle_cls=True, lang=PADDLE_OCR_LANG)
                logger.info("PaddleOCR initialized")
            except ImportError:
                logger.warning("PaddleOCR not available, falling back to Tesseract")
                self._init_tesseract()
        else:
            self._init_tesseract()
    
    def _init_tesseract(self):
        """Initialize Tesseract OCR"""
        try:
            import pytesseract
            from PIL import Image
            self.ocr_engine = "tesseract"
            logger.info("Tesseract OCR initialized")
        except ImportError:
            logger.warning("Tesseract not available, OCR will be disabled")
            self.ocr_engine = None
    
    def generate_caption(self, image: Image.Image, max_new_tokens: int = 256) -> str:
        """
        Generate Vietnamese caption for image using Qwen2VL
        
        Args:
            image: PIL Image object
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Vietnamese caption string
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": "Hãy mô tả chi tiết hình ảnh này bằng tiếng Việt. Bao gồm các đối tượng, hành động, bối cảnh, và các chi tiết quan trọng khác. Nếu nhận thấy đây là công trình hay đặc điểm văn hóa ở Việt Nam, hãy nêu tên công trình hoặc đặc điểm văn hóa đó bên cạnh mô tả hình ảnh.",
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
            logger.error(f"Error generating caption: {e}")
            return "Không thể tạo mô tả cho hình ảnh này."
    
    def extract_ocr(self, image: Image.Image) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text string
        """
        if self.ocr_engine is None:
            return ""
        
        try:
            if USE_PADDLE_OCR and isinstance(self.ocr_engine, type):
                # PaddleOCR
                result = self.ocr_engine.ocr(image, cls=True)
                if result and result[0]:
                    texts = [line[1][0] for line in result[0]]
                    return " ".join(texts)
                return ""
            else:
                # Tesseract
                import pytesseract
                text = pytesseract.image_to_string(image, lang='vie')
                return text.strip()
                
        except Exception as e:
            logger.error(f"Error in OCR: {e}")
            return ""
    
    def process_image(self, image: Image.Image) -> Dict[str, str]:
        """
        Process image to get both caption and OCR text
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with 'caption' and 'ocr' keys
        """
        logger.info("Processing image: generating caption and OCR")
        
        caption = self.generate_caption(image)
        ocr_text = self.extract_ocr(image)
        
        return {
            "caption": caption,
            "ocr": ocr_text
        }

