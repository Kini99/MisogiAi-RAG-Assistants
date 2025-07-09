"""
RAGAS evaluation pipeline for batch evaluation and real-time monitoring.
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .metrics import MedicalRAGASMetrics
from medical_rag.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGASEvaluationPipeline:
    """RAGAS evaluation pipeline for medical RAG systems."""
    
    def __init__(self):
        self.metrics = MedicalRAGASMetrics()
        self.results_dir = Path(settings.evaluation_results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Thresholds for quality control
        self.thresholds = {
            "faithfulness": settings.ragas_faithfulness_threshold,
            "context_precision": settings.ragas_context_precision_threshold,
            "context_utilization": settings.ragas_context_precision_threshold,  # Use same threshold for utilization
            "context_recall": settings.ragas_context_recall_threshold,
            "answer_relevancy": settings.ragas_answer_relevancy_threshold
        }
        
        logger.info(f"Initialized RAGAS Evaluation Pipeline with thresholds: {self.thresholds}")
    
    def evaluate_batch(
        self, 
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: Optional[List[str]] = None,
        batch_name: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of medical queries using RAGAS metrics.
        
        Args:
            questions: List of medical questions
            contexts: List of context lists for each question
            answers: List of generated answers
            ground_truths: Optional ground truth answers
            batch_name: Name for the evaluation batch
            
        Returns:
            Comprehensive evaluation results
        """
        try:
            start_time = time.time()
            
            if batch_name is None:
                batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Starting batch evaluation: {batch_name} with {len(questions)} queries")
            
            # Run RAGAS evaluation
            ragas_scores = self.metrics.evaluate_rag_system(
                questions, contexts, answers, ground_truths
            )
            
            # Calculate custom medical metrics
            custom_metrics = self.metrics.calculate_custom_medical_metrics(
                questions, contexts, answers, ground_truths
            )
            
            # Combine all metrics
            all_metrics = {**ragas_scores, **custom_metrics}
            
            # Check quality thresholds
            quality_check = self._check_quality_thresholds(all_metrics)
            
            # Calculate evaluation time
            evaluation_time = time.time() - start_time
            
            # Prepare results
            results = {
                "batch_name": batch_name,
                "timestamp": datetime.now().isoformat(),
                "evaluation_time": evaluation_time,
                "num_queries": len(questions),
                "metrics": all_metrics,
                "quality_check": quality_check,
                "thresholds": self.thresholds,
                "details": {
                    "questions": questions,
                    "contexts": contexts,
                    "answers": answers,
                    "ground_truths": ground_truths
                }
            }
            
            # Save results
            self._save_evaluation_results(results, batch_name)
            
            logger.info(f"Batch evaluation completed: {batch_name} in {evaluation_time:.2f}s")
            logger.info(f"Quality check passed: {quality_check['overall_pass']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch evaluation: {e}")
            raise
    
    def evaluate_single_query(
        self, 
        question: str,
        context: List[str],
        answer: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single medical query.
        
        Args:
            question: Medical question
            context: Context documents
            answer: Generated answer
            ground_truth: Optional ground truth answer
            
        Returns:
            Evaluation results for single query
        """
        try:
            # Run evaluation on single query
            results = self.evaluate_batch(
                [question], [context], [answer], 
                [ground_truth] if ground_truth else None,
                f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Extract single query results
            single_results = {
                "question": question,
                "context": context,
                "answer": answer,
                "ground_truth": ground_truth,
                "metrics": results["metrics"],
                "quality_check": results["quality_check"],
                "timestamp": results["timestamp"]
            }
            
            return single_results
            
        except Exception as e:
            logger.error(f"Error in single query evaluation: {e}")
            raise
    
    async def evaluate_stream(
        self, 
        query_stream: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a stream of queries asynchronously.
        
        Args:
            query_stream: List of query dictionaries with question, context, answer
            
        Returns:
            List of evaluation results
        """
        try:
            logger.info(f"Starting stream evaluation with {len(query_stream)} queries")
            
            # Prepare data for batch evaluation
            questions = [q["question"] for q in query_stream]
            contexts = [q["context"] for q in query_stream]
            answers = [q["answer"] for q in query_stream]
            ground_truths = [q.get("ground_truth") for q in query_stream]
            
            # Run batch evaluation
            results = self.evaluate_batch(
                questions, contexts, answers, ground_truths,
                f"stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Prepare individual results
            individual_results = []
            for i, query in enumerate(query_stream):
                individual_result = {
                    "query_id": i,
                    "question": query["question"],
                    "context": query["context"],
                    "answer": query["answer"],
                    "ground_truth": query.get("ground_truth"),
                    "metrics": {
                        metric: results["metrics"][metric] 
                        for metric in results["metrics"]
                    },
                    "quality_check": results["quality_check"],
                    "timestamp": results["timestamp"]
                }
                individual_results.append(individual_result)
            
            return individual_results
            
        except Exception as e:
            logger.error(f"Error in stream evaluation: {e}")
            raise
    
    def evaluate_with_retrieval(
        self, 
        questions: List[str],
        retrieval_results: List[List[Tuple[str, float]]],
        answers: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system with retrieval results.
        
        Args:
            questions: List of questions
            retrieval_results: List of (document, score) tuples for each question
            answers: Generated answers
            ground_truths: Optional ground truth answers
            
        Returns:
            Evaluation results with retrieval analysis
        """
        try:
            # Extract context documents from retrieval results
            contexts = []
            for retrieval_result in retrieval_results:
                context_docs = [doc for doc, score in retrieval_result]
                contexts.append(context_docs)
            
            # Run standard evaluation
            results = self.evaluate_batch(questions, contexts, answers, ground_truths)
            
            # Add retrieval-specific analysis
            retrieval_analysis = self._analyze_retrieval_quality(
                questions, retrieval_results, answers
            )
            
            results["retrieval_analysis"] = retrieval_analysis
            
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieval evaluation: {e}")
            raise
    
    def _check_quality_thresholds(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Check if metrics meet quality thresholds.
        
        Args:
            metrics: Dictionary of metric scores
            
        Returns:
            Quality check results
        """
        try:
            quality_check = {
                "thresholds_met": {},
                "overall_pass": True,
                "failed_metrics": [],
                "warnings": []
            }
            
            for metric, threshold in self.thresholds.items():
                if metric in metrics:
                    score = metrics[metric]
                    meets_threshold = score >= threshold
                    quality_check["thresholds_met"][metric] = {
                        "score": score,
                        "threshold": threshold,
                        "pass": meets_threshold
                    }
                    
                    if not meets_threshold:
                        quality_check["overall_pass"] = False
                        quality_check["failed_metrics"].append(metric)
                        
                        if metric == "faithfulness":
                            quality_check["warnings"].append(
                                f"Critical: Faithfulness below threshold ({score:.3f} < {threshold})"
                            )
                        else:
                            quality_check["warnings"].append(
                                f"Warning: {metric} below threshold ({score:.3f} < {threshold})"
                            )
                elif metric == "context_precision" and "context_utilization" in metrics:
                    # Handle case where context_utilization is used instead of context_precision
                    score = metrics["context_utilization"]
                    meets_threshold = score >= threshold
                    quality_check["thresholds_met"]["context_utilization"] = {
                        "score": score,
                        "threshold": threshold,
                        "pass": meets_threshold
                    }
                    
                    if not meets_threshold:
                        quality_check["overall_pass"] = False
                        quality_check["failed_metrics"].append("context_utilization")
                        quality_check["warnings"].append(
                            f"Warning: context_utilization below threshold ({score:.3f} < {threshold})"
                        )
            
            return quality_check
            
        except Exception as e:
            logger.error(f"Error checking quality thresholds: {e}")
            return {
                "thresholds_met": {},
                "overall_pass": False,
                "failed_metrics": [],
                "warnings": [f"Error in quality check: {e}"]
            }
    
    def _analyze_retrieval_quality(
        self, 
        questions: List[str],
        retrieval_results: List[List[Tuple[str, float]]],
        answers: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze retrieval quality and relevance.
        
        Args:
            questions: List of questions
            retrieval_results: Retrieval results for each question
            answers: Generated answers
            
        Returns:
            Retrieval analysis results
        """
        try:
            analysis = {
                "avg_retrieval_scores": [],
                "retrieval_coverage": [],
                "query_retrieval_correlation": []
            }
            
            for i, (question, retrieval_result, answer) in enumerate(
                zip(questions, retrieval_results, answers)
            ):
                if retrieval_result:
                    # Average retrieval score
                    avg_score = sum(score for _, score in retrieval_result) / len(retrieval_result)
                    analysis["avg_retrieval_scores"].append(avg_score)
                    
                    # Retrieval coverage (how much of answer comes from retrieved docs)
                    retrieved_text = " ".join([doc for doc, _ in retrieval_result]).lower()
                    answer_lower = answer.lower()
                    
                    # Simple coverage calculation
                    answer_words = set(answer_lower.split())
                    retrieved_words = set(retrieved_text.split())
                    
                    if answer_words:
                        coverage = len(answer_words.intersection(retrieved_words)) / len(answer_words)
                    else:
                        coverage = 0.0
                    
                    analysis["retrieval_coverage"].append(coverage)
                    
                    # Correlation between retrieval score and answer quality
                    # This is a simplified correlation
                    analysis["query_retrieval_correlation"].append(avg_score * coverage)
            
            # Calculate averages
            analysis["avg_retrieval_score"] = sum(analysis["avg_retrieval_scores"]) / len(analysis["avg_retrieval_scores"]) if analysis["avg_retrieval_scores"] else 0.0
            analysis["avg_coverage"] = sum(analysis["retrieval_coverage"]) / len(analysis["retrieval_coverage"]) if analysis["retrieval_coverage"] else 0.0
            analysis["avg_correlation"] = sum(analysis["query_retrieval_correlation"]) / len(analysis["query_retrieval_correlation"]) if analysis["query_retrieval_correlation"] else 0.0
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing retrieval quality: {e}")
            return {"error": str(e)}
    
    def _save_evaluation_results(self, results: Dict[str, Any], batch_name: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            batch_name: Name of the evaluation batch
        """
        try:
            # Create results file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{batch_name}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            # Save results
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Saved evaluation results to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
    
    def load_evaluation_results(self, batch_name: str) -> Optional[Dict[str, Any]]:
        """
        Load evaluation results from file.
        
        Args:
            batch_name: Name of the evaluation batch
            
        Returns:
            Evaluation results if found, None otherwise
        """
        try:
            # Find the most recent file for this batch
            pattern = f"{batch_name}_*.json"
            files = list(self.results_dir.glob(pattern))
            
            if not files:
                return None
            
            # Get the most recent file
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                results = json.load(f)
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading evaluation results: {e}")
            return None
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all evaluation results.
        
        Returns:
            Summary of evaluation results
        """
        try:
            summary = {
                "total_evaluations": 0,
                "avg_metrics": {},
                "quality_pass_rate": 0.0,
                "recent_evaluations": []
            }
            
            # Load all evaluation files
            json_files = list(self.results_dir.glob("*.json"))
            
            if not json_files:
                return summary
            
            all_metrics = []
            quality_passes = 0
            
            for file_path in json_files:
                try:
                    with open(file_path, 'r') as f:
                        results = json.load(f)
                    
                    summary["total_evaluations"] += 1
                    
                    if "metrics" in results:
                        all_metrics.append(results["metrics"])
                    
                    if "quality_check" in results and results["quality_check"].get("overall_pass", False):
                        quality_passes += 1
                    
                    # Add to recent evaluations
                    if len(summary["recent_evaluations"]) < 10:
                        summary["recent_evaluations"].append({
                            "batch_name": results.get("batch_name", "Unknown"),
                            "timestamp": results.get("timestamp", ""),
                            "num_queries": results.get("num_queries", 0),
                            "quality_pass": results.get("quality_check", {}).get("overall_pass", False)
                        })
                        
                except Exception as e:
                    logger.warning(f"Error loading evaluation file {file_path}: {e}")
                    continue
            
            # Calculate averages
            if all_metrics:
                metric_names = all_metrics[0].keys()
                for metric in metric_names:
                    values = [m.get(metric, 0.0) for m in all_metrics if m.get(metric) is not None]
                    if values:
                        summary["avg_metrics"][metric] = sum(values) / len(values)
            
            # Calculate quality pass rate
            if summary["total_evaluations"] > 0:
                summary["quality_pass_rate"] = quality_passes / summary["total_evaluations"]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating evaluation summary: {e}")
            return {"error": str(e)} 