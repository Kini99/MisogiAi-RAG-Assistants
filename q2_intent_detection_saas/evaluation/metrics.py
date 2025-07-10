"""
Metrics Calculator for Evaluation
Calculates intent accuracy, response relevance, context utilization, and other metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class MetricsCalculator:
    """Calculator for various evaluation metrics"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Expected response patterns for each intent
        self.expected_patterns = {
            "technical": [
                "api", "code", "error", "authentication", "documentation",
                "integration", "setup", "configuration", "troubleshooting"
            ],
            "billing": [
                "pricing", "plan", "subscription", "payment", "billing",
                "cost", "refund", "upgrade", "downgrade", "invoice"
            ],
            "feature": [
                "feature", "request", "roadmap", "timeline", "voting",
                "portal", "planned", "future", "enhancement", "improvement"
            ]
        }
    
    def calculate_intent_accuracy(self, expected_intents: List[str], predicted_intents: List[str]) -> Dict[str, float]:
        """Calculate intent classification accuracy metrics"""
        
        # Overall accuracy
        accuracy = accuracy_score(expected_intents, predicted_intents)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            expected_intents, predicted_intents, average=None, labels=['technical', 'billing', 'feature']
        )
        
        # Confusion matrix
        cm = confusion_matrix(expected_intents, predicted_intents, labels=['technical', 'billing', 'feature'])
        
        # Per-intent accuracy
        intent_accuracy = {}
        for i, intent in enumerate(['technical', 'billing', 'feature']):
            intent_accuracy[intent] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
        
        return {
            'overall_accuracy': accuracy,
            'per_intent_metrics': intent_accuracy,
            'confusion_matrix': cm.tolist(),
            'macro_avg_precision': np.mean(precision),
            'macro_avg_recall': np.mean(recall),
            'macro_avg_f1': np.mean(f1)
        }
    
    def calculate_response_relevance(self, queries: List[str], responses: List[str], expected_intents: List[str]) -> Dict[str, float]:
        """Calculate response relevance using cosine similarity"""
        
        # Combine queries and responses for vectorization
        combined_texts = queries + responses
        
        # Fit and transform
        try:
            tfidf_matrix = self.vectorizer.fit_transform(combined_texts)
            
            # Split back into queries and responses
            n_queries = len(queries)
            query_vectors = tfidf_matrix[:n_queries]
            response_vectors = tfidf_matrix[n_queries:]
            
            # Calculate cosine similarity
            similarities = []
            for i in range(len(queries)):
                similarity = cosine_similarity(query_vectors[i:i+1], response_vectors[i:i+1])[0][0]
                similarities.append(similarity)
            
            # Calculate relevance by intent
            intent_relevance = {}
            for intent in ['technical', 'billing', 'feature']:
                intent_indices = [i for i, exp_intent in enumerate(expected_intents) if exp_intent == intent]
                if intent_indices:
                    intent_similarities = [similarities[i] for i in intent_indices]
                    intent_relevance[intent] = {
                        'mean': np.mean(intent_similarities),
                        'std': np.std(intent_similarities),
                        'min': np.min(intent_similarities),
                        'max': np.max(intent_similarities)
                    }
            
            return {
                'overall_mean_relevance': np.mean(similarities),
                'overall_std_relevance': np.std(similarities),
                'per_intent_relevance': intent_relevance,
                'individual_similarities': similarities
            }
            
        except Exception as e:
            # Fallback to simple word overlap
            return self._calculate_word_overlap_relevance(queries, responses, expected_intents)
    
    def _calculate_word_overlap_relevance(self, queries: List[str], responses: List[str], expected_intents: List[str]) -> Dict[str, float]:
        """Fallback relevance calculation using word overlap"""
        
        similarities = []
        for query, response in zip(queries, responses):
            query_words = set(re.findall(r'\w+', query.lower()))
            response_words = set(re.findall(r'\w+', response.lower()))
            
            if query_words and response_words:
                overlap = len(query_words.intersection(response_words))
                union = len(query_words.union(response_words))
                similarity = overlap / union if union > 0 else 0
            else:
                similarity = 0
            
            similarities.append(similarity)
        
        # Calculate relevance by intent
        intent_relevance = {}
        for intent in ['technical', 'billing', 'feature']:
            intent_indices = [i for i, exp_intent in enumerate(expected_intents) if exp_intent == intent]
            if intent_indices:
                intent_similarities = [similarities[i] for i in intent_indices]
                intent_relevance[intent] = {
                    'mean': np.mean(intent_similarities),
                    'std': np.std(intent_similarities),
                    'min': np.min(intent_similarities),
                    'max': np.max(intent_similarities)
                }
        
        return {
            'overall_mean_relevance': np.mean(similarities),
            'overall_std_relevance': np.std(similarities),
            'per_intent_relevance': intent_relevance,
            'individual_similarities': similarities,
            'method': 'word_overlap'
        }
    
    def calculate_context_utilization(self, responses: List[str], expected_intents: List[str]) -> Dict[str, float]:
        """Calculate how well responses utilize context for each intent"""
        
        utilization_scores = []
        intent_utilization = {'technical': [], 'billing': [], 'feature': []}
        
        for response, expected_intent in zip(responses, expected_intents):
            # Check for expected patterns in response
            expected_patterns = self.expected_patterns.get(expected_intent, [])
            response_lower = response.lower()
            
            # Count how many expected patterns are present
            pattern_matches = sum(1 for pattern in expected_patterns if pattern in response_lower)
            
            # Calculate utilization score (0-1)
            if expected_patterns:
                utilization_score = pattern_matches / len(expected_patterns)
            else:
                utilization_score = 0
            
            utilization_scores.append(utilization_score)
            intent_utilization[expected_intent].append(utilization_score)
        
        # Calculate per-intent utilization
        per_intent_utilization = {}
        for intent in ['technical', 'billing', 'feature']:
            if intent_utilization[intent]:
                per_intent_utilization[intent] = {
                    'mean': np.mean(intent_utilization[intent]),
                    'std': np.std(intent_utilization[intent]),
                    'min': np.min(intent_utilization[intent]),
                    'max': np.max(intent_utilization[intent])
                }
        
        return {
            'overall_mean_utilization': np.mean(utilization_scores),
            'overall_std_utilization': np.std(utilization_scores),
            'per_intent_utilization': per_intent_utilization,
            'individual_scores': utilization_scores
        }
    
    def calculate_response_quality_metrics(self, responses: List[str]) -> Dict[str, float]:
        """Calculate general response quality metrics"""
        
        quality_metrics = {
            'length': [],
            'readability': [],
            'structure': [],
            'completeness': []
        }
        
        for response in responses:
            # Response length
            words = response.split()
            quality_metrics['length'].append(len(words))
            
            # Readability (simplified Flesch Reading Ease)
            sentences = response.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            quality_metrics['readability'].append(1.0 / (1.0 + avg_sentence_length / 20.0))  # Simplified
            
            # Structure (presence of formatting)
            has_structure = any(marker in response for marker in ['**', '*', '-', '1.', '2.', '3.'])
            quality_metrics['structure'].append(1.0 if has_structure else 0.0)
            
            # Completeness (has multiple sentences)
            quality_metrics['completeness'].append(1.0 if len(sentences) > 2 else 0.5)
        
        # Calculate averages
        return {
            'avg_length': np.mean(quality_metrics['length']),
            'avg_readability': np.mean(quality_metrics['readability']),
            'avg_structure': np.mean(quality_metrics['structure']),
            'avg_completeness': np.mean(quality_metrics['completeness']),
            'individual_metrics': quality_metrics
        }
    
    def calculate_performance_metrics(self, response_times: List[float], token_usage: List[int]) -> Dict[str, float]:
        """Calculate performance metrics"""
        
        return {
            'avg_response_time': np.mean(response_times),
            'std_response_time': np.std(response_times),
            'min_response_time': np.min(response_times),
            'max_response_time': np.max(response_times),
            'avg_tokens': np.mean(token_usage) if token_usage else 0,
            'total_tokens': np.sum(token_usage) if token_usage else 0,
            'queries_per_second': 1.0 / np.mean(response_times) if response_times else 0
        }
    
    def calculate_ab_test_metrics(self, local_results: Dict[str, Any], openai_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate A/B test metrics between local and OpenAI models"""
        
        # Compare key metrics
        comparison = {
            'accuracy_comparison': {
                'local': local_results.get('intent_accuracy', {}).get('overall_accuracy', 0),
                'openai': openai_results.get('intent_accuracy', {}).get('overall_accuracy', 0),
                'difference': openai_results.get('intent_accuracy', {}).get('overall_accuracy', 0) - 
                            local_results.get('intent_accuracy', {}).get('overall_accuracy', 0)
            },
            'relevance_comparison': {
                'local': local_results.get('response_relevance', {}).get('overall_mean_relevance', 0),
                'openai': openai_results.get('response_relevance', {}).get('overall_mean_relevance', 0),
                'difference': openai_results.get('response_relevance', {}).get('overall_mean_relevance', 0) - 
                            local_results.get('response_relevance', {}).get('overall_mean_relevance', 0)
            },
            'performance_comparison': {
                'local_avg_time': local_results.get('performance_metrics', {}).get('avg_response_time', 0),
                'openai_avg_time': openai_results.get('performance_metrics', {}).get('avg_response_time', 0),
                'speed_improvement': local_results.get('performance_metrics', {}).get('avg_response_time', 0) / 
                                  max(openai_results.get('performance_metrics', {}).get('avg_response_time', 1), 1)
            },
            'cost_comparison': {
                'local_tokens': local_results.get('performance_metrics', {}).get('total_tokens', 0),
                'openai_tokens': openai_results.get('performance_metrics', {}).get('total_tokens', 0),
                'token_efficiency': local_results.get('performance_metrics', {}).get('total_tokens', 0) / 
                                  max(openai_results.get('performance_metrics', {}).get('total_tokens', 1), 1)
            }
        }
        
        # Determine winner
        local_score = 0
        openai_score = 0
        
        if comparison['accuracy_comparison']['difference'] > 0:
            openai_score += 1
        else:
            local_score += 1
        
        if comparison['relevance_comparison']['difference'] > 0:
            openai_score += 1
        else:
            local_score += 1
        
        if comparison['performance_comparison']['speed_improvement'] > 1:
            local_score += 1
        else:
            openai_score += 1
        
        comparison['winner'] = 'openai' if openai_score > local_score else 'local' if local_score > openai_score else 'tie'
        comparison['scores'] = {'local': local_score, 'openai': openai_score}
        
        return comparison
    
    def generate_evaluation_report(self, all_metrics: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report"""
        
        report = "# Customer Support System Evaluation Report\n\n"
        
        # Intent Accuracy
        intent_acc = all_metrics.get('intent_accuracy', {})
        report += f"## Intent Classification Accuracy\n"
        report += f"- Overall Accuracy: {intent_acc.get('overall_accuracy', 0):.3f}\n"
        report += f"- Macro Average F1: {intent_acc.get('macro_avg_f1', 0):.3f}\n\n"
        
        # Response Relevance
        relevance = all_metrics.get('response_relevance', {})
        report += f"## Response Relevance\n"
        report += f"- Overall Mean Relevance: {relevance.get('overall_mean_relevance', 0):.3f}\n"
        report += f"- Overall Std Relevance: {relevance.get('overall_std_relevance', 0):.3f}\n\n"
        
        # Context Utilization
        utilization = all_metrics.get('context_utilization', {})
        report += f"## Context Utilization\n"
        report += f"- Overall Mean Utilization: {utilization.get('overall_mean_utilization', 0):.3f}\n"
        report += f"- Overall Std Utilization: {utilization.get('overall_std_utilization', 0):.3f}\n\n"
        
        # Performance
        performance = all_metrics.get('performance_metrics', {})
        report += f"## Performance Metrics\n"
        report += f"- Average Response Time: {performance.get('avg_response_time', 0):.3f}s\n"
        report += f"- Queries per Second: {performance.get('queries_per_second', 0):.3f}\n"
        report += f"- Total Tokens Used: {performance.get('total_tokens', 0)}\n\n"
        
        # A/B Test Results
        if 'ab_test_metrics' in all_metrics:
            ab_test = all_metrics['ab_test_metrics']
            report += f"## A/B Test Results\n"
            report += f"- Winner: {ab_test.get('winner', 'N/A')}\n"
            report += f"- Local Score: {ab_test.get('scores', {}).get('local', 0)}\n"
            report += f"- OpenAI Score: {ab_test.get('scores', {}).get('openai', 0)}\n\n"
        
        return report 