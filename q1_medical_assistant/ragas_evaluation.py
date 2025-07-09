#!/usr/bin/env python3
"""
Standalone RAGAS evaluation script for medical RAG systems.

This script provides batch evaluation capabilities for medical RAG systems
using RAGAS metrics including faithfulness, context precision, context recall,
and answer relevancy.
"""

import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from ragas_framework.evaluation import RAGASEvaluationPipeline
from ragas_framework.metrics import MedicalRAGASMetrics
from medical_rag.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
settings = get_settings()


def load_evaluation_dataset(file_path: str) -> Dict[str, List[str]]:
    """
    Load evaluation dataset from file.
    
    Args:
        file_path: Path to dataset file (CSV or JSON)
        
    Returns:
        Dictionary with questions, contexts, answers, and ground_truths
    """
    try:
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Validate required columns
        required_columns = ['question', 'context', 'answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Extract data
        dataset = {
            'questions': df['question'].tolist(),
            'contexts': df['context'].apply(lambda x: [x] if isinstance(x, str) else x).tolist(),
            'answers': df['answer'].tolist(),
            'ground_truths': df.get('ground_truth', [None] * len(df)).tolist()
        }
        
        logger.info(f"Loaded dataset with {len(dataset['questions'])} samples")
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def create_sample_dataset() -> Dict[str, List[str]]:
    """
    Create a sample medical evaluation dataset.
    
    Returns:
        Sample dataset for testing
    """
    sample_data = {
        'questions': [
            "What are the common side effects of aspirin?",
            "How should diabetes be managed?",
            "What is the recommended dosage for ibuprofen?",
            "What are the symptoms of hypertension?",
            "How does penicillin work as an antibiotic?"
        ],
        'contexts': [
            ["Aspirin is a common pain reliever that can cause stomach upset, bleeding, and allergic reactions in some patients."],
            ["Diabetes management involves blood sugar monitoring, diet control, exercise, and medication as prescribed by healthcare providers."],
            ["Ibuprofen dosage typically ranges from 200-800mg every 4-6 hours, but should be taken as directed by a doctor."],
            ["Hypertension symptoms may include headaches, shortness of breath, nosebleeds, and chest pain, though many patients are asymptomatic."],
            ["Penicillin works by interfering with bacterial cell wall synthesis, preventing bacteria from growing and reproducing."]
        ],
        'answers': [
            "Aspirin can cause stomach upset, bleeding, and allergic reactions. Always consult your healthcare provider before taking any medication.",
            "Diabetes management includes monitoring blood sugar, following a healthy diet, regular exercise, and taking prescribed medications under medical supervision.",
            "Ibuprofen dosage is typically 200-800mg every 4-6 hours, but you should always follow your doctor's specific instructions.",
            "Hypertension symptoms can include headaches, shortness of breath, nosebleeds, and chest pain, though many people have no symptoms.",
            "Penicillin works by preventing bacteria from building their cell walls, which stops them from growing and reproducing."
        ],
        'ground_truths': [
            "Aspirin side effects include stomach irritation, bleeding risk, and potential allergic reactions.",
            "Diabetes management requires blood sugar monitoring, dietary control, exercise, and medication adherence.",
            "Ibuprofen dosage varies but typically ranges from 200-800mg every 4-6 hours as prescribed.",
            "Hypertension symptoms include headaches, shortness of breath, nosebleeds, and chest pain.",
            "Penicillin inhibits bacterial cell wall synthesis to prevent bacterial growth and reproduction."
        ]
    }
    
    return sample_data


def run_evaluation(
    dataset: Dict[str, List[str]],
    batch_name: Optional[str] = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation on the dataset.
    
    Args:
        dataset: Evaluation dataset
        batch_name: Name for the evaluation batch
        save_results: Whether to save results to file
        
    Returns:
        Evaluation results
    """
    try:
        evaluation_pipeline = RAGASEvaluationPipeline()
        
        logger.info(f"Starting evaluation with {len(dataset['questions'])} samples")
        
        # Run evaluation
        results = evaluation_pipeline.evaluate_batch(
            questions=dataset['questions'],
            contexts=dataset['contexts'],
            answers=dataset['answers'],
            ground_truths=dataset['ground_truths'],
            batch_name=batch_name
        )
        
        # Print results
        print("\n" + "="*60)
        print("RAGAS EVALUATION RESULTS")
        print("="*60)
        
        print(f"Batch Name: {results['batch_name']}")
        print(f"Evaluation Time: {results['evaluation_time']:.2f} seconds")
        print(f"Number of Queries: {results['num_queries']}")
        
        print("\nMETRICS:")
        print("-" * 30)
        for metric, score in results['metrics'].items():
            print(f"{metric.replace('_', ' ').title()}: {score:.3f}")
        
        print("\nQUALITY CHECK:")
        print("-" * 30)
        quality_check = results['quality_check']
        print(f"Overall Pass: {quality_check['overall_pass']}")
        
        if quality_check['failed_metrics']:
            print(f"Failed Metrics: {', '.join(quality_check['failed_metrics'])}")
        
        if quality_check['warnings']:
            print("Warnings:")
            for warning in quality_check['warnings']:
                print(f"  - {warning}")
        
        print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        raise


def run_individual_metrics_evaluation(dataset: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Run individual RAGAS metrics evaluation.
    
    Args:
        dataset: Evaluation dataset
        
    Returns:
        Individual metric scores
    """
    try:
        metrics = MedicalRAGASMetrics()
        
        logger.info("Running individual metrics evaluation")
        
        # Context Precision (or Context Utilization if no ground truth)
        context_precision = metrics.evaluate_context_precision(
            dataset['questions'],
            dataset['contexts'],
            dataset['ground_truths']
        )
        
        # Context Recall (requires ground truth)
        context_recall = metrics.evaluate_context_recall(
            dataset['questions'],
            dataset['contexts'],
            dataset['ground_truths']
        )
        
        # Faithfulness
        faithfulness = metrics.evaluate_faithfulness(
            dataset['questions'],
            dataset['contexts'],
            dataset['answers']
        )
        
        # Answer Relevancy
        answer_relevancy = metrics.evaluate_answer_relevancy(
            dataset['questions'],
            dataset['answers']
        )
        
        # Custom medical metrics
        custom_metrics = metrics.calculate_custom_medical_metrics(
            dataset['questions'],
            dataset['contexts'],
            dataset['answers'],
            dataset['ground_truths']
        )
        
        # Determine the correct metric name based on ground truth availability
        has_ground_truth = dataset['ground_truths'] and any(gt for gt in dataset['ground_truths'] if gt)
        
        results = {
            'context_precision' if has_ground_truth else 'context_utilization': context_precision,
            'context_recall': context_recall,
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            **custom_metrics
        }
        
        print("\nINDIVIDUAL METRICS EVALUATION:")
        print("-" * 40)
        for metric, score in results.items():
            print(f"{metric.replace('_', ' ').title()}: {score:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in individual metrics evaluation: {e}")
        raise


def main():
    """Main function for RAGAS evaluation script."""
    parser = argparse.ArgumentParser(
        description="RAGAS evaluation for medical RAG systems"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to evaluation dataset file (CSV or JSON)'
    )
    
    parser.add_argument(
        '--batch-name',
        type=str,
        help='Name for the evaluation batch'
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use sample dataset for evaluation'
    )
    
    parser.add_argument(
        '--individual-metrics',
        action='store_true',
        help='Run individual metrics evaluation'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save evaluation results'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    try:
        # Load dataset
        if args.sample:
            logger.info("Using sample dataset")
            dataset = create_sample_dataset()
        elif args.dataset:
            logger.info(f"Loading dataset from {args.dataset}")
            dataset = load_evaluation_dataset(args.dataset)
        else:
            logger.info("No dataset specified, using sample dataset")
            dataset = create_sample_dataset()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run evaluation
        start_time = time.time()
        
        if args.individual_metrics:
            results = run_individual_metrics_evaluation(dataset)
        else:
            results = run_evaluation(
                dataset,
                batch_name=args.batch_name,
                save_results=not args.no_save
            )
        
        total_time = time.time() - start_time
        
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        
        # Save results if requested
        if not args.no_save and 'batch_name' in results:
            output_file = output_dir / f"{results['batch_name']}_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {output_file}")
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 