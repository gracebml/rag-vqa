"""
Gradio Demo App for RAG-enhanced VQA
"""
import gradio as gr
import logging
from PIL import Image
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import RAGVQAPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize pipeline
logger.info("Loading pipeline...")
pipeline = RAGVQAPipeline(use_4bit=True)
logger.info("Pipeline loaded successfully")


def process_vqa(image: Image.Image, question: str) -> tuple:
    """
    Process VQA query and return answer with intermediate results
    
    Args:
        image: Uploaded image
        question: User question
        
    Returns:
        Tuple of (answer, caption, ocr, context)
    """
    if image is None:
        return "Vui lòng upload hình ảnh.", "", "", ""
    
    if not question or not question.strip():
        return "Vui lòng nhập câu hỏi.", "", "", ""
    
    try:
        result = pipeline.process(
            image=image,
            question=question.strip(),
            return_intermediate=True
        )
        
        answer = result["answer"]
        caption = result.get("caption", "")
        ocr = result.get("ocr", "")
        context = result.get("context", "")
        
        return answer, caption, ocr, context
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"Lỗi: {str(e)}", "", "", ""


# Create Gradio interface
with gr.Blocks(title="RAG-Enhanced VQA - Lịch sử & Văn hóa Việt Nam") as demo:
    gr.Markdown(
        """
        # RAG-Enhanced Visual Question Answering
        ## Hệ thống VQA về Lịch sử và Văn hóa Việt Nam
        
        Hệ thống này sử dụng:
        - **Qwen2VL-7B** để hiểu hình ảnh và tạo câu trả lời
        - **RAG (Retrieval Augmented Generation)** để bổ sung kiến thức từ database lịch sử
        - **OCR** để đọc văn bản trong hình ảnh (bia đá, câu đối, v.v.)
        
        ### Hướng dẫn sử dụng:
        1. Upload một hình ảnh liên quan đến lịch sử/văn hóa Việt Nam
        2. Nhập câu hỏi bằng tiếng Việt
        3. Nhấn "Trả lời" để xem kết quả
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload hình ảnh",
                height=400
            )
            question_input = gr.Textbox(
                label="Câu hỏi",
                placeholder="Ví dụ: Đây là gì? Ý nghĩa lịch sử của hình ảnh này là gì?",
                lines=3
            )
            submit_btn = gr.Button("Trả lời", variant="primary")
        
        with gr.Column(scale=1):
            answer_output = gr.Textbox(
                label="Câu trả lời",
                lines=10,
                interactive=False
            )
    
    with gr.Accordion("Chi tiết xử lý (Intermediate Results)", open=False):
        with gr.Row():
            caption_output = gr.Textbox(
                label="Mô tả hình ảnh (Caption)",
                lines=5,
                interactive=False
            )
            ocr_output = gr.Textbox(
                label="Văn bản trong ảnh (OCR)",
                lines=5,
                interactive=False
            )
        context_output = gr.Textbox(
            label="Kiến thức được sử dụng (Retrieved Context)",
            lines=10,
            interactive=False
        )
    
    # Example questions
    gr.Markdown("### Ví dụ câu hỏi:")
    examples = gr.Examples(
        examples=[
            ["Đây là gì?", "Hình ảnh này thể hiện điều gì?"],
            ["Ý nghĩa văn hóa của hình ảnh này là gì?", "Bối cảnh lịch sử của hình ảnh này là gì?"],
            ["Hãy giải thích chi tiết về hình ảnh này.", "Thông tin trong hình ảnh này nói về gì?"]
        ],
        inputs=question_input
    )
    
    # Connect interface
    submit_btn.click(
        fn=process_vqa,
        inputs=[image_input, question_input],
        outputs=[answer_output, caption_output, ocr_output, context_output]
    )
    
    # Allow Enter key to submit
    question_input.submit(
        fn=process_vqa,
        inputs=[image_input, question_input],
        outputs=[answer_output, caption_output, ocr_output, context_output]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

