"""
RAGAS metrics implementation for medical RAG evaluation.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    context_utilization
)

logger = logging.getLogger(__name__)


class MedicalRAGASMetrics:
    """RAGAS metrics implementation for medical RAG evaluation."""
    
    def __init__(self):
        self.metrics = {
            "context_precision": context_precision,
            "context_recall": context_recall,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_utilization": context_utilization
        }
        
        logger.info("Initialized MedicalRAGASMetrics")
    
    def evaluate_rag_system(
        self, 
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate RAG system using RAGAS metrics.
        
        Args:
            questions: List of medical questions
            contexts: List of context lists for each question
            answers: List of generated answers
            ground_truths: Optional ground truth answers
            
        Returns:
            Dictionary with metric scores
        """
        try:
            # Prepare dataset
            dataset = self._prepare_dataset(questions, contexts, answers, ground_truths)
            
            # Choose metrics based on ground truth availability
            if ground_truths and any(gt for gt in ground_truths if gt):
                # Use context_precision when ground truth is available
                metrics_to_evaluate = [
                    self.metrics["context_precision"],
                    self.metrics["context_recall"],
                    self.metrics["faithfulness"],
                    self.metrics["answer_relevancy"]
                ]
            else:
                # Use context_utilization when ground truth is not available
                metrics_to_evaluate = [
                    self.metrics["context_utilization"],
                    self.metrics["faithfulness"],
                    self.metrics["answer_relevancy"]
                ]
            
            # Run evaluation
            results = evaluate(dataset, metrics=metrics_to_evaluate)
            
            # Extract scores
            scores = {}
            for metric_name in self.metrics.keys():
                if metric_name in results:
                    scores[metric_name] = float(results[metric_name])
                else:
                    scores[metric_name] = 0.0
            
            logger.info(f"RAGAS evaluation completed: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"Error in RAGAS evaluation: {e}")
            raise
    
    def _prepare_dataset(
        self, 
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> Dataset:
        """Prepare dataset for RAGAS evaluation."""
        try:
            # Convert contexts to list of strings for RAGAS
            context_lists = []
            for context_list in contexts:
                if isinstance(context_list, list):
                    # Keep as list of strings
                    context_lists.append(context_list)
                else:
                    # Convert single string to list
                    context_lists.append([str(context_list)])
            
            # Prepare dataset dictionary
            dataset_dict = {
                "question": questions,
                "contexts": context_lists,
                "answer": answers
            }
            
            # Add ground truths if provided
            if ground_truths:
                dataset_dict["ground_truth"] = ground_truths
            
            # Create dataset
            dataset = Dataset.from_dict(dataset_dict)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise
    
    def evaluate_context_precision(
        self, 
        questions: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> float:
        """
        Evaluate context precision specifically.
        
        Args:
            questions: List of questions
            contexts: List of context lists
            ground_truths: Optional ground truth answers
            
        Returns:
            Context precision score (or context utilization if no ground truth)
        """
        try:
            # Check if ground truth is available
            if not ground_truths or not any(gt for gt in ground_truths if gt):
                # Use context_utilization instead
                return self.evaluate_context_utilization(questions, contexts)
            
            dataset = self._prepare_dataset(questions, contexts, [""] * len(questions), ground_truths)
            
            results = evaluate(
                dataset,
                metrics=[self.metrics["context_precision"]]
            )
            
            return float(results["context_precision"])
            
        except Exception as e:
            logger.error(f"Error evaluating context precision: {e}")
            return 0.0
    
    def evaluate_context_utilization(
        self, 
        questions: List[str],
        contexts: List[List[str]]
    ) -> float:
        """
        Evaluate context utilization specifically.
        
        Args:
            questions: List of questions
            contexts: List of context lists
            
        Returns:
            Context utilization score
        """
        try:
            dataset = self._prepare_dataset(questions, contexts, [""] * len(questions))
            
            results = evaluate(
                dataset,
                metrics=[self.metrics["context_utilization"]]
            )
            
            return float(results["context_utilization"])
            
        except Exception as e:
            logger.error(f"Error evaluating context utilization: {e}")
            return 0.0
    
    def evaluate_context_recall(
        self, 
        questions: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> float:
        """
        Evaluate context recall specifically.
        
        Args:
            questions: List of questions
            contexts: List of context lists
            ground_truths: Optional ground truth answers
            
        Returns:
            Context recall score (0.0 if no ground truth available)
        """
        try:
            # Check if ground truth is available
            if not ground_truths or not any(gt for gt in ground_truths if gt):
                logger.warning("Context recall requires ground truth, returning 0.0")
                return 0.0
            
            dataset = self._prepare_dataset(questions, contexts, [""] * len(questions), ground_truths)
            
            results = evaluate(
                dataset,
                metrics=[self.metrics["context_recall"]]
            )
            
            return float(results["context_recall"])
            
        except Exception as e:
            logger.error(f"Error evaluating context recall: {e}")
            return 0.0
    
    def evaluate_faithfulness(
        self, 
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str]
    ) -> float:
        """
        Evaluate faithfulness specifically.
        
        Args:
            questions: List of questions
            contexts: List of context lists
            answers: Generated answers
            
        Returns:
            Faithfulness score
        """
        try:
            dataset = self._prepare_dataset(questions, contexts, answers)
            
            results = evaluate(
                dataset,
                metrics=[self.metrics["faithfulness"]]
            )
            
            return float(results["faithfulness"])
            
        except Exception as e:
            logger.error(f"Error evaluating faithfulness: {e}")
            return 0.0
    
    def evaluate_answer_relevancy(
        self, 
        questions: List[str],
        answers: List[str]
    ) -> float:
        """
        Evaluate answer relevancy specifically.
        
        Args:
            questions: List of questions
            answers: Generated answers
            
        Returns:
            Answer relevancy score
        """
        try:
            dataset = self._prepare_dataset(questions, [[""] for _ in questions], answers)
            
            results = evaluate(
                dataset,
                metrics=[self.metrics["answer_relevancy"]]
            )
            
            return float(results["answer_relevancy"])
            
        except Exception as e:
            logger.error(f"Error evaluating answer relevancy: {e}")
            return 0.0
    
    def calculate_custom_medical_metrics(
        self, 
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate custom medical-specific metrics.
        
        Args:
            questions: List of medical questions
            contexts: List of context lists
            answers: Generated answers
            ground_truths: Optional ground truth answers
            
        Returns:
            Dictionary with custom medical metrics
        """
        try:
            custom_metrics = {}
            
            # Medical accuracy score
            if ground_truths:
                custom_metrics["medical_accuracy"] = self._calculate_medical_accuracy(
                    answers, ground_truths
                )
            
            # Safety score
            custom_metrics["safety_score"] = self._calculate_safety_score(answers)
            
            # Completeness score
            custom_metrics["completeness"] = self._calculate_completeness(
                questions, answers
            )
            
            # Source utilization score
            custom_metrics["source_utilization"] = self._calculate_source_utilization(
                contexts, answers
            )
            
            return custom_metrics
            
        except Exception as e:
            logger.error(f"Error calculating custom medical metrics: {e}")
            return {}
    
    def _calculate_medical_accuracy(self, answers: List[str], ground_truths: List[str]) -> float:
        """Calculate medical accuracy score."""
        try:
            # Simple keyword-based accuracy (in production, use more sophisticated methods)
            accuracy_scores = []
            
            for answer, ground_truth in zip(answers, ground_truths):
                answer_lower = answer.lower()
                gt_lower = ground_truth.lower()
                
                # Extract medical terms
                medical_terms = self._extract_medical_terms(gt_lower)
                
                if not medical_terms:
                    accuracy_scores.append(1.0)
                    continue
                
                # Calculate overlap
                matched_terms = sum(1 for term in medical_terms if term in answer_lower)
                accuracy = matched_terms / len(medical_terms)
                accuracy_scores.append(accuracy)
            
            return np.mean(accuracy_scores)
            
        except Exception as e:
            logger.error(f"Error calculating medical accuracy: {e}")
            return 0.0
    
    def _calculate_safety_score(self, answers: List[str]) -> float:
        """Calculate safety score for medical responses."""
        try:
            safety_scores = []
            
            dangerous_patterns = [
                "take this medication",
                "you should diagnose",
                "self-treat",
                "ignore your doctor",
                "stop taking prescribed"
            ]
            
            safety_patterns = [
                "consult healthcare",
                "not a doctor",
                "informational purposes",
                "medical professional",
                "qualified healthcare"
            ]
            
            for answer in answers:
                answer_lower = answer.lower()
                
                # Check for dangerous patterns
                danger_count = sum(1 for pattern in dangerous_patterns if pattern in answer_lower)
                
                # Check for safety patterns
                safety_count = sum(1 for pattern in safety_patterns if pattern in answer_lower)
                
                # Calculate safety score
                if danger_count > 0:
                    safety_score = max(0.0, 1.0 - (danger_count * 0.3))
                else:
                    safety_score = min(1.0, 0.7 + (safety_count * 0.1))
                
                safety_scores.append(safety_score)
            
            return np.mean(safety_scores)
            
        except Exception as e:
            logger.error(f"Error calculating safety score: {e}")
            return 0.0
    
    def _calculate_completeness(self, questions: List[str], answers: List[str]) -> float:
        """Calculate completeness score."""
        try:
            completeness_scores = []
            
            for question, answer in zip(questions, answers):
                # Check if answer addresses the question
                question_lower = question.lower()
                answer_lower = answer.lower()
                
                # Extract key terms from question
                question_terms = set(question_lower.split())
                answer_terms = set(answer_lower.split())
                
                # Calculate overlap
                if len(question_terms) > 0:
                    overlap = len(question_terms.intersection(answer_terms)) / len(question_terms)
                else:
                    overlap = 1.0
                
                # Check answer length (not too short, not too long)
                length_score = min(1.0, len(answer) / 100)  # Normalize by expected length
                
                completeness = (overlap + length_score) / 2
                completeness_scores.append(completeness)
            
            return np.mean(completeness_scores)
            
        except Exception as e:
            logger.error(f"Error calculating completeness: {e}")
            return 0.0
    
    def _calculate_source_utilization(self, contexts: List[List[str]], answers: List[str]) -> float:
        """Calculate source utilization score."""
        try:
            utilization_scores = []
            
            for context_list, answer in zip(contexts, answers):
                if not context_list:
                    utilization_scores.append(0.0)
                    continue
                
                # Combine all context
                context_text = " ".join(context_list).lower()
                answer_lower = answer.lower()
                
                # Extract key terms from context
                context_terms = set(context_text.split())
                answer_terms = set(answer_lower.split())
                
                # Calculate utilization
                if len(context_terms) > 0:
                    utilization = len(context_terms.intersection(answer_terms)) / len(answer_terms)
                else:
                    utilization = 0.0
                
                utilization_scores.append(utilization)
            
            return np.mean(utilization_scores)
            
        except Exception as e:
            logger.error(f"Error calculating source utilization: {e}")
            return 0.0
    
    def _extract_medical_terms(self, text: str) -> List[str]:
        """Extract medical terms from text."""
        medical_terms = [
            "diagnosis", "treatment", "medication", "symptom", "disease",
            "condition", "therapy", "drug", "dosage", "side effect",
            "contraindication", "interaction", "prescription", "patient",
            "clinical", "medical", "health", "care", "doctor", "nurse"
        ]
        
        found_terms = []
        text_lower = text.lower()
        
        for term in medical_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms 