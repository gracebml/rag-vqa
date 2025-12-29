"""
Script để test module RAG retrieval locally
"""
import sys
from pathlib import Path
import logging
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval import RetrievalModule
from src.vision import VisionModule
from src.config import KB_JSON_PATH, DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_retrieval_without_vlm():
    """Test retrieval module without VLM (chỉ test BM25/Embedding search)"""
    print("\n" + "="*60)
    print("TEST 1: Retrieval Module (without VLM)")
    print("="*60)
    
    # Initialize retrieval module (không cần VLM)
    retrieval = RetrievalModule()
    
    # Test queries
    test_queries = [
        "Văn Miếu là gì?",
        "Lịch sử Hoàng thành Thăng Long",
        "Chùa Một Cột ở đâu?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 60)
        
        results = retrieval.retrieve(
            query=query,
            caption="",
            ocr_text=""
        )
        
        print(f"✅ Retrieved {len(results)} documents:")
        for i, doc in enumerate(results, 1):
            print(f"\n  [{i}] Source: {doc.get('source', 'unknown')}")
            if 'entity' in doc:
                print(f"      Entity: {doc.get('entity', '')}")
                print(f"      Summary: {doc.get('summary', '')[:100]}...")
            elif 'title' in doc:
                print(f"      Title: {doc.get('title', '')}")
                print(f"      Content: {doc.get('content', '')[:100]}...")
            if 'score' in doc:
                print(f"      Score: {doc.get('score', 0):.4f}")


def test_retrieval_with_vlm():
    """Test retrieval module with VLM (cần GPU và model)"""
    print("\n" + "="*60)
    print("TEST 2: Retrieval Module with VLM (cần GPU)")
    print("="*60)
    
    try:
        # Initialize vision module (cần GPU)
        print("Loading Vision Module (this may take a few minutes)...")
        vision_module = VisionModule(use_4bit=True)
        
        # Initialize retrieval with vision module
        retrieval = RetrievalModule(vision_module=vision_module)
        
        # Test với ảnh thật
        images_dir = DATA_DIR / "images_flat"
        if images_dir.exists():
            # Tìm một ảnh để test
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            if image_files:
                test_image_path = image_files[0]
                print(f"\n  Using image: {test_image_path.name}")
                
                image = Image.open(test_image_path).convert('RGB')
                
                # Generate caption
                print("Generating caption...")
                vision_results = vision_module.process_image(image)
                caption = vision_results["caption"]
                ocr_text = vision_results["ocr"]
                
                print(f"Caption: {caption[:150]}...")
                if ocr_text:
                    print(f"OCR: {ocr_text[:100]}...")
                
                # Test retrieval với image
                query = "Đây là gì? Hãy mô tả chi tiết."
                print(f"\n Query: {query}")
                print("-" * 60)
                
                results = retrieval.retrieve(
                    query=query,
                    caption=caption,
                    ocr_text=ocr_text,
                    image=image
                )
                
                print(f" Retrieved {len(results)} documents:")
                for i, doc in enumerate(results, 1):
                    print(f"\n  [{i}] Source: {doc.get('source', 'unknown')}")
                    if 'entity' in doc:
                        print(f"      Entity: {doc.get('entity', '')}")
                    elif 'title' in doc:
                        print(f"      Title: {doc.get('title', '')}")
                        if 'keywords_used' in doc:
                            print(f"      Keywords used: {doc.get('keywords_used', '')}")
                    print(f"      Content preview: {doc.get('content', doc.get('summary', ''))[:150]}...")
            else:
                print("  No images found in images_flat directory")
        else:
            print("  Images directory not found, skipping image test")
            
    except Exception as e:
        print(f" Error: {e}")
        print("  VLM test requires GPU. Skipping...")


def test_wikipedia_search():
    """Test Wikipedia search functionality"""
    print("\n" + "="*60)
    print("TEST 3: Wikipedia Search")
    print("="*60)
    
    retrieval = RetrievalModule()
    
    if retrieval.wikipedia is None:
        print("  Wikipedia not available. Install: pip install wikipedia")
        return
    
    test_queries = [
        "Văn Miếu Quốc Tử Giám",
        "Hoàng thành Thăng Long"
    ]
    
    for query in test_queries:
        print(f"\n Query: {query}")
        print("-" * 60)
        
        results = retrieval._wikipedia_search(query, max_results=2)
        
        print(f" Found {len(results)} Wikipedia pages:")
        for i, doc in enumerate(results, 1):
            print(f"\n  [{i}] Title: {doc.get('title', '')}")
            print(f"      URL: {doc.get('url', '')}")
            print(f"      Content: {doc.get('content', '')[:200]}...")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("RAG RETRIEVAL MODULE TEST")
    print("="*60)
    
    # Check if knowledge base exists
    if not KB_JSON_PATH.exists():
        print(f"  Warning: Knowledge base not found at {KB_JSON_PATH}")
        print("   Some tests may not work properly.")
        print("   Please ensure knowledge_base.json is in the data directory.")
    
    # Test 1: Basic retrieval (không cần GPU)
    test_retrieval_without_vlm()
    
    # Test 2: Wikipedia search
    test_wikipedia_search()
    
    # Test 3: Retrieval with VLM (cần GPU)
    print("\n" + "="*60)
    print("Testing with VLM (requires GPU)...")
    print("="*60)
    user_input = input("Do you want to test with VLM? This requires GPU. (y/n): ")
    if user_input.lower() == 'y':
        test_retrieval_with_vlm()
    else:
        print("Skipping VLM test.")
    
    print("\n" + "="*60)
    print(" Testing completed!")
    print("="*60)


if __name__ == "__main__":
    main()

