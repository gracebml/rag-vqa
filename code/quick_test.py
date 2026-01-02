

import os
from evaluation_metrics import BERTScoreEvaluator, LLMJudgeEvaluator, VQAEvaluator

def test_bert_score():
    """Test BERTScore evaluator"""
    print("="*80)
    print("Testing BERTScore Evaluator")
    print("="*80)
    
    evaluator = BERTScoreEvaluator(device="cuda")
    
    # Test cases
    test_cases = [
        {
            "name": "Good answer",
            "prediction": "Chùa Một Cột được xây dựng năm 1049 dưới triều Lý.",
            "ground_truth": "Chùa Một Cột, triều Lý, năm 1049."
        },
        {
            "name": "Wrong answer",
            "prediction": "Đây là Văn Miếu, được xây dựng năm 1070.",
            "ground_truth": "Chùa Một Cột, triều Lý, năm 1049."
        },
        {
            "name": "Partially correct",
            "prediction": "Vịnh Hạ Long là một di sản nổi tiếng.",
            "ground_truth": "Vịnh Hạ Long là di sản thiên nhiên thế giới, UNESCO công nhận 1994."
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        score = evaluator.evaluate_single(case["prediction"], case["ground_truth"])
        print(f"\nTest {i}: {case['name']}")
        print(f"  Prediction: {case['prediction']}")
        print(f"  Ground Truth: {case['ground_truth']}")
        print(f"  BERTScore: {score:.4f}")
    
    print("\n BERTScore test completed!")


def test_llm_judge():
    """Test LLM Judge evaluator"""
    print("Testing LLM Judge Evaluator")
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: GEMINI_API_KEY not set!")
        print("Please set your API key:")
        print("  export GEMINI_API_KEY='your-key-here'")
        print("\nGet free API key at: https://makersuite.google.com/app/apikey")
        return
    
    evaluator = LLMJudgeEvaluator()
    
    # Test case
    question = "Địa điểm trong ảnh có phải di sản thiên nhiên thế giới không?"
    prediction = "Có, đây là Vịnh Hạ Long, một di sản thiên nhiên thế giới được UNESCO công nhận năm 1994."
    ground_truth = "Có, Vịnh Hạ Long là di sản thiên nhiên thế giới, được UNESCO công nhận năm 1994."
    
    print(f"\nQuestion: {question}")
    print(f"\nPrediction: {prediction}")
    print(f"\nGround Truth: {ground_truth}")
    print("\n⏳ Calling Gemini API...")
    
    score, reasoning = evaluator.evaluate_single(question, prediction, ground_truth)
    
    print(f"\n Score: {score}/5")
    print(f" Reasoning: {reasoning}")
    
    print("\n LLM Judge test completed!")


def test_combined():
    """Test combined evaluator"""
    print("Testing Combined Evaluator")
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n  Skipping combined test (no API key)")
        return
    
    evaluator = VQAEvaluator(
        use_bert_score=True,
        use_llm_judge=True,
        device="cuda"
    )
    
    result = evaluator.evaluate_single(
        image_id="test_001",
        question="Đây là di tích gì?",
        prediction="Văn Miếu - Quốc Tử Giám, được xây dựng năm 1070.",
        ground_truth="Văn Miếu - Quốc Tử Giám, thành lập năm 1070 dưới triều Lý."
    )
    
    print(f"\n Results for {result.image_id}:")
    print(f"   Question: {result.question}")
    print(f"\n   Prediction: {result.predicted_answer}")
    print(f"   Ground Truth: {result.ground_truth}")
    print(f"\n Metrics:")
    print(f"   - BERTScore: {result.bert_score:.4f}")
    print(f"   - LLM Judge: {result.llm_judge_score}/5")
    print(f"\nLLM Reasoning: {result.llm_judge_reasoning}")
    
    print("\n Combined test completed!")


def main():
    """Run all tests"""
    print("EVALUATION METRICS QUICK TEST")
    
    # Test 1: BERTScore (no API key needed)
    try:
        test_bert_score()
    except Exception as e:
        print(f"\n❌ BERTScore test failed: {e}")
    
    # Test 2: LLM Judge (needs API key)
    try:
        test_llm_judge()
    except Exception as e:
        print(f"\n❌ LLM Judge test failed: {e}")
    
    # Test 3: Combined
    try:
        test_combined()
    except Exception as e:
        print(f"\n❌ Combined test failed: {e}")
    
    print("ALL TESTS COMPLETED!")
    print("\n Next steps:")
    print("  1. Set GEMINI_API_KEY if you haven't: export GEMINI_API_KEY='your-key'")
    print("  2. Check README_EVALUATION.md for detailed usage")
    print("  3. Run on full dataset: python evaluation_metrics.py --predictions data.json --output results.json")
    print()


if __name__ == "__main__":
    main()
