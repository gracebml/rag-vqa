"""
Evaluation Metrics for Vietnamese Cultural VQA


This module implements two main evaluation metrics:
1. BERTScore: Semantic similarity between predicted and ground truth answers
2. LLM-as-a-Judge: Using Gemini Flash API to score answers on 1-5 scale
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

# BERTScore imports
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("Warning: bert_score not installed. Install with: pip install bert-score")
    BERTSCORE_AVAILABLE = False

# Gemini API imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")
    GEMINI_AVAILABLE = False


@dataclass
class EvaluationResult:
    """Store evaluation results for a single sample"""
    image_id: str
    question: str
    predicted_answer: str
    ground_truth: str
    bert_score: Optional[float] = None
    llm_judge_score: Optional[float] = None
    llm_judge_reasoning: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "image_id": self.image_id,
            "question": self.question,
            "predicted_answer": self.predicted_answer,
            "ground_truth": self.ground_truth,
            "bert_score": self.bert_score,
            "llm_judge_score": self.llm_judge_score,
            "llm_judge_reasoning": self.llm_judge_reasoning
        }


class BERTScoreEvaluator:
    """
    Evaluate answers using BERTScore metric
    BERTScore computes semantic similarity between predicted and reference texts
    using contextual embeddings
    """
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased", device: str = "cuda"):
        """
        Initialize BERTScore evaluator
        
        Args:
            model_name: Name of the BERT model to use for scoring
            device: Device to run evaluation on ('cuda' or 'cpu')
        """
        if not BERTSCORE_AVAILABLE:
            raise ImportError("bert-score package not installed")
        
        self.model_name = model_name
        self.device = device
        print(f"Initialized BERTScore with model: {model_name}")
    
    def evaluate_single(self, prediction: str, reference: str) -> float:
        """
        Evaluate a single prediction-reference pair
        
        Args:
            prediction: Model's predicted answer
            reference: Ground truth answer
            
        Returns:
            F1 score from BERTScore (0-1 scale)
        """
        P, R, F1 = bert_score(
            [prediction], 
            [reference],
            model_type=self.model_name,
            device=self.device,
            lang="other",  # For multilingual models
            verbose=False
        )
        return F1.item()
    
    def evaluate_batch(self, predictions: List[str], references: List[str]) -> List[float]:
        """
        Evaluate multiple predictions at once (more efficient)
        
        Args:
            predictions: List of predicted answers
            references: List of ground truth answers
            
        Returns:
            List of F1 scores
        """
        P, R, F1 = bert_score(
            predictions,
            references,
            model_type=self.model_name,
            device=self.device,
            lang="other",
            verbose=False
        )
        return F1.tolist()
    
    def get_statistics(self, scores: List[float]) -> Dict[str, float]:
        """
        Calculate statistics for a list of scores
        
        Args:
            scores: List of BERTScore F1 values
            
        Returns:
            Dictionary with mean, std, min, max statistics
        """
        scores_array = np.array(scores)
        return {
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "median": float(np.median(scores_array))
        }


class LLMJudgeEvaluator:
    """
    Evaluate answers using LLM-as-a-Judge approach with Gemini Flash API
    Following the methodology from LaVy paper
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """
        Initialize LLM Judge evaluator
        
        Args:
            api_key: Gemini API key (if None, will try to get from env)
            model_name: Gemini model to use (default: gemini-1.5-flash for free tier)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")
        
        # Get API key
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Gemini API key not provided. Either pass api_key parameter or set GEMINI_API_KEY environment variable.\n"
                    "Get your free API key at: https://makersuite.google.com/app/apikey"
                )
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"Initialized LLM Judge with model: {model_name}")
        
        # System prompt for judging (Vietnamese Cultural VQA specific)
        self.judge_prompt_template = """Bạn là một chuyên gia đánh giá hệ thống VQA (Visual Question Answering) về lịch sử và văn hóa Việt Nam.

Nhiệm vụ của bạn là đánh giá chất lượng câu trả lời của mô hình so với câu trả lời chuẩn (ground truth).

**Tiêu chí đánh giá (thang điểm 1-5):**

**5 điểm (Xuất sắc):**
- Câu trả lời chính xác hoàn toàn về mặt lịch sử/văn hóa
- Cung cấp đầy đủ thông tin như ground truth
- Có thể bổ sung thêm chi tiết hữu ích và chính xác
- Ngôn ngữ rõ ràng, mạch lạc

**4 điểm (Tốt):**
- Câu trả lời chính xác về các thông tin quan trọng
- Có thể thiếu một vài chi tiết nhỏ không ảnh hưởng nhiều
- Ngôn ngữ rõ ràng

**3 điểm (Trung bình):**
- Câu trả lời đúng một phần
- Có thể thiếu một số thông tin quan trọng
- Hoặc có một vài sai sót nhỏ về chi tiết

**2 điểm (Yếu):**
- Câu trả lời sai về thông tin quan trọng
- Hoặc thiếu nhiều thông tin cần thiết
- Có thể có hallucination (thông tin sai hoặc bịa đặt)

**1 điểm (Rất yếu):**
- Câu trả lời hoàn toàn sai
- Không liên quan đến câu hỏi
- Chứa nhiều thông tin sai lệch nghiêm trọng

---

**Câu hỏi:** {question}

**Câu trả lời của mô hình:** {prediction}

**Câu trả lời chuẩn (Ground Truth):** {ground_truth}

---

**Yêu cầu:**
1. Đánh giá câu trả lời của mô hình theo thang điểm 1-5
2. Giải thích ngắn gọn lý do chấm điểm đó (2-3 câu)

**Định dạng trả lời:**
Điểm: [số điểm]
Lý do: [giải thích ngắn gọn]
"""
    
    def evaluate_single(self, question: str, prediction: str, ground_truth: str) -> Tuple[float, str]:
        """
        Evaluate a single answer using LLM Judge
        
        Args:
            question: The original question
            prediction: Model's predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            Tuple of (score, reasoning)
        """
        # Format prompt
        prompt = self.judge_prompt_template.format(
            question=question,
            prediction=prediction,
            ground_truth=ground_truth
        )
        
        try:
            # Generate evaluation
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse response
            score, reasoning = self._parse_response(response_text)
            return score, reasoning
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return None, f"Error: {str(e)}"
    
    def _parse_response(self, response_text: str) -> Tuple[float, str]:
        """
        Parse LLM response to extract score and reasoning
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Tuple of (score, reasoning)
        """
        lines = response_text.strip().split('\n')
        score = None
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            
            # Extract score
            if line.startswith("Điểm:") or line.startswith("Score:"):
                score_str = line.split(":")[-1].strip()
                # Extract number from string (handle cases like "Điểm: 4/5" or "Điểm: 4 điểm")
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', score_str)
                if match:
                    score = float(match.group(1))
            
            # Extract reasoning
            elif line.startswith("Lý do:") or line.startswith("Reason:"):
                reasoning = line.split(":", 1)[-1].strip()
            
            # If reasoning spans multiple lines
            elif reasoning and line and not line.startswith("Điểm") and not line.startswith("Score"):
                reasoning += " " + line
        
        # Validate score
        if score is None:
            print(f"Warning: Could not parse score from response: {response_text}")
            score = 0.0
        elif score < 1 or score > 5:
            print(f"Warning: Invalid score {score}. Clamping to [1, 5]")
            score = max(1, min(5, score))
        
        return score, reasoning
    
    def evaluate_batch(self, questions: List[str], predictions: List[str], 
                      ground_truths: List[str], delay: float = 1.0) -> List[Tuple[float, str]]:
        """
        Evaluate multiple answers with rate limiting
        
        Args:
            questions: List of questions
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            delay: Delay between API calls in seconds (to avoid rate limits)
            
        Returns:
            List of (score, reasoning) tuples
        """
        import time
        
        results = []
        for q, p, gt in tqdm(zip(questions, predictions, ground_truths), 
                            total=len(questions), 
                            desc="LLM Judge Evaluation"):
            score, reasoning = self.evaluate_single(q, p, gt)
            results.append((score, reasoning))
            
            # Rate limiting
            time.sleep(delay)
        
        return results
    
    def get_statistics(self, scores: List[float]) -> Dict[str, float]:
        """
        Calculate statistics for LLM Judge scores
        
        Args:
            scores: List of scores (1-5 scale)
            
        Returns:
            Dictionary with statistics
        """
        scores_array = np.array([s for s in scores if s is not None])
        
        if len(scores_array) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0
            }
        
        return {
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "median": float(np.median(scores_array))
        }


class VQAEvaluator:
    """
    Combined evaluator for VQA using both BERTScore and LLM Judge
    """
    
    def __init__(self, 
                 use_bert_score: bool = True,
                 use_llm_judge: bool = True,
                 gemini_api_key: Optional[str] = None,
                 bert_model: str = "bert-base-multilingual-cased",
                 device: str = "cuda"):
        """
        Initialize combined evaluator
        
        Args:
            use_bert_score: Whether to use BERTScore
            use_llm_judge: Whether to use LLM Judge
            gemini_api_key: API key for Gemini (required if use_llm_judge=True)
            bert_model: Model for BERTScore
            device: Device for computation
        """
        self.use_bert_score = use_bert_score
        self.use_llm_judge = use_llm_judge
        
        # Initialize evaluators
        if use_bert_score:
            self.bert_evaluator = BERTScoreEvaluator(model_name=bert_model, device=device)
        
        if use_llm_judge:
            self.llm_evaluator = LLMJudgeEvaluator(api_key=gemini_api_key)
    
    def evaluate_single(self, 
                       image_id: str,
                       question: str, 
                       prediction: str, 
                       ground_truth: str) -> EvaluationResult:
        """
        Evaluate a single VQA sample
        
        Args:
            image_id: ID of the image
            question: The question
            prediction: Model's answer
            ground_truth: Correct answer
            
        Returns:
            EvaluationResult object
        """
        result = EvaluationResult(
            image_id=image_id,
            question=question,
            predicted_answer=prediction,
            ground_truth=ground_truth
        )
        
        # BERTScore evaluation
        if self.use_bert_score:
            result.bert_score = self.bert_evaluator.evaluate_single(prediction, ground_truth)
        
        # LLM Judge evaluation
        if self.use_llm_judge:
            score, reasoning = self.llm_evaluator.evaluate_single(question, prediction, ground_truth)
            result.llm_judge_score = score
            result.llm_judge_reasoning = reasoning
        
        return result
    
    def evaluate_dataset(self, 
                        predictions_file: str,
                        output_file: str,
                        llm_judge_delay: float = 1.0) -> Dict:
        """
        Evaluate entire dataset and save results
        
        Args:
            predictions_file: JSON file with predictions in format:
                [{"image_id": "...", "question": "...", "prediction": "...", "ground_truth": "..."}, ...]
            output_file: Path to save evaluation results
            llm_judge_delay: Delay between LLM API calls
            
        Returns:
            Dictionary with overall statistics
        """
        # Load predictions
        print(f"Loading predictions from {predictions_file}...")
        with open(predictions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} samples")
        
        # Prepare data
        image_ids = [item["image_id"] for item in data]
        questions = [item["question"] for item in data]
        predictions = [item["prediction"] for item in data]
        ground_truths = [item["ground_truth"] for item in data]
        
        results = []
        
        # BERTScore evaluation (batch mode for efficiency)
        bert_scores = None
        if self.use_bert_score:
            print("\n" + "="*80)
            print("Running BERTScore evaluation...")
            print("="*80)
            bert_scores = self.bert_evaluator.evaluate_batch(predictions, ground_truths)
            bert_stats = self.bert_evaluator.get_statistics(bert_scores)
            print(f"\nBERTScore Statistics:")
            print(f"  Mean: {bert_stats['mean']:.4f}")
            print(f"  Std:  {bert_stats['std']:.4f}")
            print(f"  Min:  {bert_stats['min']:.4f}")
            print(f"  Max:  {bert_stats['max']:.4f}")
        
        # LLM Judge evaluation (with progress bar)
        llm_scores = None
        llm_reasonings = None
        if self.use_llm_judge:
            print("\n" + "="*80)
            print("Running LLM Judge evaluation...")
            print("="*80)
            llm_results = self.llm_evaluator.evaluate_batch(
                questions, predictions, ground_truths, delay=llm_judge_delay
            )
            llm_scores = [r[0] for r in llm_results]
            llm_reasonings = [r[1] for r in llm_results]
            
            llm_stats = self.llm_evaluator.get_statistics(llm_scores)
            print(f"\nLLM Judge Statistics:")
            print(f"  Mean: {llm_stats['mean']:.4f}")
            print(f"  Std:  {llm_stats['std']:.4f}")
            print(f"  Min:  {llm_stats['min']:.4f}")
            print(f"  Max:  {llm_stats['max']:.4f}")
        
        # Combine results
        for i in range(len(data)):
            result = EvaluationResult(
                image_id=image_ids[i],
                question=questions[i],
                predicted_answer=predictions[i],
                ground_truth=ground_truths[i],
                bert_score=bert_scores[i] if bert_scores else None,
                llm_judge_score=llm_scores[i] if llm_scores else None,
                llm_judge_reasoning=llm_reasonings[i] if llm_reasonings else None
            )
            results.append(result)
        
        # Save results
        print(f"\nSaving results to {output_file}...")
        output_data = {
            "results": [r.to_dict() for r in results],
            "statistics": {}
        }
        
        if self.use_bert_score:
            output_data["statistics"]["bert_score"] = bert_stats
        
        if self.use_llm_judge:
            output_data["statistics"]["llm_judge"] = llm_stats
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Results saved!")
        
        return output_data["statistics"]


def main():
    """
    Example usage of the evaluation metrics
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions")
    parser.add_argument("--predictions", type=str, required=True,
                       help="Path to predictions JSON file")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save evaluation results")
    parser.add_argument("--gemini-api-key", type=str, default=None,
                       help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--no-bert-score", action="store_true",
                       help="Disable BERTScore evaluation")
    parser.add_argument("--no-llm-judge", action="store_true",
                       help="Disable LLM Judge evaluation")
    parser.add_argument("--bert-model", type=str, 
                       default="bert-base-multilingual-cased",
                       help="BERT model for BERTScore")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--llm-delay", type=float, default=1.0,
                       help="Delay between LLM API calls (seconds)")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = VQAEvaluator(
        use_bert_score=not args.no_bert_score,
        use_llm_judge=not args.no_llm_judge,
        gemini_api_key=args.gemini_api_key,
        bert_model=args.bert_model,
        device=args.device
    )
    
    # Run evaluation
    stats = evaluator.evaluate_dataset(
        predictions_file=args.predictions,
        output_file=args.output,
        llm_judge_delay=args.llm_delay
    )
    
    # Print summary
    print("EVALUATION COMPLETE")
    print("\nFinal Statistics:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
