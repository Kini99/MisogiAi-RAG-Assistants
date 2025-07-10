"""
Main Evaluator for Customer Support System
Runs comprehensive evaluation tests and generates reports
"""

import time
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from support_system import CustomerSupportSystem
from .test_queries import TestQueryGenerator
from .metrics import MetricsCalculator

logger = logging.getLogger(__name__)

class Evaluator:
    """Main evaluator for the customer support system"""
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        self.support_system = CustomerSupportSystem()
        self.test_generator = TestQueryGenerator()
        self.metrics_calculator = MetricsCalculator()
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluation results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_queries': [],
            'local_results': {},
            'openai_results': {},
            'ab_test_results': {},
            'summary': {}
        }
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation with all test queries"""
        logger.info("Starting full evaluation...")
        
        # Get all test queries
        all_queries = self.test_generator.get_all_test_queries()
        expected_intents = [self.test_generator.get_expected_intent(q) for q in all_queries]
        
        logger.info(f"Testing {len(all_queries)} queries across all intents")
        
        # Test with local model (processor-based approach)
        logger.info("Testing with local model (processor-based)...")
        local_results = self._evaluate_queries(all_queries, expected_intents, use_llm_direct=False)
        
        # Test with OpenAI model (direct LLM approach)
        logger.info("Testing with OpenAI model (direct LLM)...")
        openai_results = self._evaluate_queries(all_queries, expected_intents, use_llm_direct=True)
        
        # Calculate A/B test metrics
        logger.info("Calculating A/B test metrics...")
        ab_test_results = self.metrics_calculator.calculate_ab_test_metrics(local_results, openai_results)
        
        # Store results
        self.results['test_queries'] = all_queries
        self.results['local_results'] = local_results
        self.results['openai_results'] = openai_results
        self.results['ab_test_results'] = ab_test_results
        
        # Generate summary
        self.results['summary'] = self._generate_summary()
        
        # Save results
        self._save_results()
        
        logger.info("Full evaluation completed")
        return self.results
    
    def run_intent_evaluation(self, intent: str) -> Dict[str, Any]:
        """Run evaluation for a specific intent"""
        logger.info(f"Starting evaluation for intent: {intent}")
        
        queries = self.test_generator.get_queries_by_intent(intent)
        expected_intents = [intent] * len(queries)
        
        logger.info(f"Testing {len(queries)} queries for {intent} intent")
        
        # Test both approaches
        local_results = self._evaluate_queries(queries, expected_intents, use_llm_direct=False)
        openai_results = self._evaluate_queries(queries, expected_intents, use_llm_direct=True)
        
        # Calculate comparison
        ab_test_results = self.metrics_calculator.calculate_ab_test_metrics(local_results, openai_results)
        
        results = {
            'intent': intent,
            'queries': queries,
            'local_results': local_results,
            'openai_results': openai_results,
            'ab_test_results': ab_test_results
        }
        
        # Save intent-specific results
        self._save_intent_results(intent, results)
        
        return results
    
    def run_balanced_evaluation(self, samples_per_intent: int = 5) -> Dict[str, Any]:
        """Run evaluation with balanced samples per intent"""
        logger.info(f"Starting balanced evaluation with {samples_per_intent} samples per intent")
        
        balanced_queries = self.test_generator.get_balanced_sample(samples_per_intent)
        expected_intents = [self.test_generator.get_expected_intent(q) for q in balanced_queries]
        
        logger.info(f"Testing {len(balanced_queries)} balanced queries")
        
        # Test both approaches
        local_results = self._evaluate_queries(balanced_queries, expected_intents, use_llm_direct=False)
        openai_results = self._evaluate_queries(balanced_queries, expected_intents, use_llm_direct=True)
        
        # Calculate comparison
        ab_test_results = self.metrics_calculator.calculate_ab_test_metrics(local_results, openai_results)
        
        results = {
            'queries': balanced_queries,
            'expected_intents': expected_intents,
            'local_results': local_results,
            'openai_results': openai_results,
            'ab_test_results': ab_test_results
        }
        
        # Save balanced results
        self._save_balanced_results(results)
        
        return results
    
    def _evaluate_queries(self, queries: List[str], expected_intents: List[str], use_llm_direct: bool = False) -> Dict[str, Any]:
        """Evaluate a list of queries and return metrics"""
        
        predicted_intents = []
        responses = []
        response_times = []
        token_usage = []
        
        logger.info(f"Processing {len(queries)} queries with {'direct LLM' if use_llm_direct else 'processor-based'} approach")
        
        for i, query in enumerate(queries):
            try:
                start_time = time.time()
                
                if use_llm_direct:
                    result = self.support_system.process_query_with_llm(query)
                else:
                    result = self.support_system.process_query(query)
                
                response_time = time.time() - start_time
                
                predicted_intents.append(result.intent.intent)
                responses.append(result.response)
                response_times.append(response_time)
                token_usage.append(result.tokens_used)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(queries)} queries")
                
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                # Add fallback values
                predicted_intents.append("technical")
                responses.append("Error processing query")
                response_times.append(30.0)  # Penalty time
                token_usage.append(0)
        
        # Calculate metrics
        intent_accuracy = self.metrics_calculator.calculate_intent_accuracy(expected_intents, predicted_intents)
        response_relevance = self.metrics_calculator.calculate_response_relevance(queries, responses, expected_intents)
        context_utilization = self.metrics_calculator.calculate_context_utilization(responses, expected_intents)
        response_quality = self.metrics_calculator.calculate_response_quality_metrics(responses)
        performance_metrics = self.metrics_calculator.calculate_performance_metrics(response_times, token_usage)
        
        return {
            'intent_accuracy': intent_accuracy,
            'response_relevance': response_relevance,
            'context_utilization': context_utilization,
            'response_quality': response_quality,
            'performance_metrics': performance_metrics,
            'raw_data': {
                'queries': queries,
                'expected_intents': expected_intents,
                'predicted_intents': predicted_intents,
                'responses': responses,
                'response_times': response_times,
                'token_usage': token_usage
            }
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary"""
        local_results = self.results['local_results']
        openai_results = self.results['openai_results']
        ab_test = self.results['ab_test_results']
        
        summary = {
            'total_queries_tested': len(self.results['test_queries']),
            'intent_distribution': self.test_generator.get_intent_distribution(),
            'overall_accuracy': {
                'local': local_results.get('intent_accuracy', {}).get('overall_accuracy', 0),
                'openai': openai_results.get('intent_accuracy', {}).get('overall_accuracy', 0)
            },
            'overall_relevance': {
                'local': local_results.get('response_relevance', {}).get('overall_mean_relevance', 0),
                'openai': openai_results.get('response_relevance', {}).get('overall_mean_relevance', 0)
            },
            'performance': {
                'local_avg_time': local_results.get('performance_metrics', {}).get('avg_response_time', 0),
                'openai_avg_time': openai_results.get('performance_metrics', {}).get('avg_response_time', 0),
                'speed_improvement': ab_test.get('performance_comparison', {}).get('speed_improvement', 0)
            },
            'ab_test_winner': ab_test.get('winner', 'unknown'),
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        local_results = self.results['local_results']
        openai_results = self.results['openai_results']
        ab_test = self.results['ab_test_results']
        
        # Accuracy recommendations
        local_acc = local_results.get('intent_accuracy', {}).get('overall_accuracy', 0)
        openai_acc = openai_results.get('intent_accuracy', {}).get('overall_accuracy', 0)
        
        if local_acc < 0.8:
            recommendations.append("Consider improving local model training data for better intent classification")
        
        if abs(local_acc - openai_acc) > 0.1:
            recommendations.append("Significant accuracy gap between local and OpenAI models - investigate prompt engineering")
        
        # Performance recommendations
        local_time = local_results.get('performance_metrics', {}).get('avg_response_time', 0)
        if local_time > 10:
            recommendations.append("Local model response time is high - consider model optimization or caching")
        
        # Relevance recommendations
        local_rel = local_results.get('response_relevance', {}).get('overall_mean_relevance', 0)
        if local_rel < 0.5:
            recommendations.append("Response relevance is low - improve prompt engineering and context utilization")
        
        # Cost recommendations
        if ab_test.get('winner') == 'local':
            recommendations.append("Local model performs well - consider using it as primary to reduce costs")
        else:
            recommendations.append("OpenAI model performs better - use local model as fallback only")
        
        return recommendations
    
    def _save_results(self):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results
        results_file = os.path.join(self.output_dir, f"evaluation_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate and save report
        report = self.metrics_calculator.generate_evaluation_report(self.results)
        report_file = os.path.join(self.output_dir, f"evaluation_report_{timestamp}.md")
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save summary
        summary_file = os.path.join(self.output_dir, f"evaluation_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(self.results['summary'], f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"Files created: {results_file}, {report_file}, {summary_file}")
    
    def _save_intent_results(self, intent: str, results: Dict[str, Any]):
        """Save intent-specific results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"{intent}_evaluation_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Intent results saved to {filename}")
    
    def _save_balanced_results(self, results: Dict[str, Any]):
        """Save balanced evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"balanced_evaluation_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Balanced results saved to {filename}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health before evaluation"""
        return self.support_system.health_check()
    
    def print_summary(self):
        """Print evaluation summary to console"""
        if not self.results.get('summary'):
            print("No evaluation results available. Run evaluation first.")
            return
        
        summary = self.results['summary']
        
        print("\n" + "="*60)
        print("CUSTOMER SUPPORT SYSTEM EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\nTotal Queries Tested: {summary['total_queries_tested']}")
        print(f"Intent Distribution: {summary['intent_distribution']}")
        
        print(f"\nIntent Classification Accuracy:")
        print(f"  Local Model: {summary['overall_accuracy']['local']:.3f}")
        print(f"  OpenAI Model: {summary['overall_accuracy']['openai']:.3f}")
        
        print(f"\nResponse Relevance:")
        print(f"  Local Model: {summary['overall_relevance']['local']:.3f}")
        print(f"  OpenAI Model: {summary['overall_relevance']['openai']:.3f}")
        
        print(f"\nPerformance (Average Response Time):")
        print(f"  Local Model: {summary['performance']['local_avg_time']:.3f}s")
        print(f"  OpenAI Model: {summary['performance']['openai_avg_time']:.3f}s")
        print(f"  Speed Improvement: {summary['performance']['speed_improvement']:.2f}x")
        
        print(f"\nA/B Test Winner: {summary['ab_test_winner'].upper()}")
        
        print(f"\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  - {rec}")
        
        print("\n" + "="*60) 