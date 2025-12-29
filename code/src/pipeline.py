"""
Main Pipeline: Combine Vision, Retrieval, and Answering modules
"""
import logging
from typing import Dict, Optional
from PIL import Image

from .vision import VisionModule
from .retrieval import RetrievalModule
from .answering import AnsweringModule

logger = logging.getLogger(__name__)


class RAGVQAPipeline:
    """
    Complete RAG-enhanced VQA Pipeline
    
    Following the framework from:
    https://towardsdatascience.com/a-simple-framework-for-rag-enhanced-visual-question-answering-06768094762e/
    """
    
    def __init__(
        self,
        vision_model_name: Optional[str] = None,
        answering_model_name: Optional[str] = None,
        use_4bit: bool = True
    ):
        """
        Initialize the complete pipeline
        
        Args:
            vision_model_name: Model name for vision module (default from config)
            answering_model_name: Model name for answering module (default from config)
            use_4bit: Whether to use 4-bit quantization for answering module
        """
        logger.info("Initializing RAG-VQA Pipeline...")
        
        # Initialize modules
        # Use default from config if None is passed
        if vision_model_name is None:
            try:
                try:
                    from .config import QWEN2VL_MODEL_NAME
                except ImportError:
                    from config import QWEN2VL_MODEL_NAME
                vision_model_name = QWEN2VL_MODEL_NAME
            except (ImportError, NameError):
                # Fallback if config import fails
                vision_model_name = "Qwen/Qwen2-VL-7B-Instruct"
                logger.warning(f"Could not import QWEN2VL_MODEL_NAME from config, using default: {vision_model_name}")
        
        if answering_model_name is None:
            try:
                try:
                    from .config import QWEN2VL_MODEL_NAME
                except ImportError:
                    from config import QWEN2VL_MODEL_NAME
                answering_model_name = QWEN2VL_MODEL_NAME
            except (ImportError, NameError):
                # Fallback if config import fails
                answering_model_name = "Qwen/Qwen2-VL-7B-Instruct"
                logger.warning(f"Could not import QWEN2VL_MODEL_NAME from config, using default: {answering_model_name}")
        
        self.vision_module = VisionModule(model_name=vision_model_name, use_4bit=use_4bit)
        # Pass vision_module to retrieval_module for VLM-based keyword generation
        self.retrieval_module = RetrievalModule(vision_module=self.vision_module)
        self.answering_module = AnsweringModule(
            model_name=answering_model_name,
            use_4bit=use_4bit
        )
        
        logger.info("Pipeline initialized successfully")
    
    def process(
        self,
        image: Image.Image,
        question: str,
        return_intermediate: bool = False
    ) -> Dict:
        """
        Process a VQA query through the complete pipeline
        
        Args:
            image: PIL Image object
            question: User question in Vietnamese
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Dictionary with answer and optionally intermediate results
        """
        logger.info(f"Processing question: {question}")
        
        # Step 1: Vision-to-Text (Caption + OCR)
        logger.info("Step 1: Vision processing...")
        vision_results = self.vision_module.process_image(image)
        caption = vision_results["caption"]
        ocr_text = vision_results["ocr"]
        
        logger.info(f"Caption: {caption[:100]}...")
        logger.info(f"OCR: {ocr_text[:100] if ocr_text else 'None'}...")
        
        # Step 2: Knowledge Retrieval (RAG)
        logger.info("Step 2: Knowledge retrieval...")
        retrieved_docs = self.retrieval_module.retrieve(
            query=question,
            caption=caption,
            ocr_text=ocr_text,
            image=image  # Pass image for VLM keyword generation in Wikipedia search
        )
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Format context
        context = self.answering_module.format_context(retrieved_docs)
        
        # Step 3: Answer Generation
        logger.info("Step 3: Answer generation...")
        answer = self.answering_module.generate_answer(
            image=image,
            question=question,
            caption=caption,
            ocr_text=ocr_text,
            context=context
        )
        
        logger.info("Processing completed")
        
        # Prepare result
        result = {
            "answer": answer,
            "question": question
        }
        
        if return_intermediate:
            result.update({
                "caption": caption,
                "ocr": ocr_text,
                "retrieved_docs": retrieved_docs,
                "context": context
            })
        
        return result
    
    def __call__(self, image: Image.Image, question: str) -> str:
        """
        Convenience method to get just the answer
        
        Args:
            image: PIL Image object
            question: User question
            
        Returns:
            Answer string
        """
        result = self.process(image, question, return_intermediate=False)
        return result["answer"]

