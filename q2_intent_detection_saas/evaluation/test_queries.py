"""
Test Query Generator
Generates test queries for evaluation (20 per intent category)
"""

from typing import Dict, List, Tuple
import random

class TestQueryGenerator:
    """Generator for test queries across all intent categories"""
    
    def __init__(self):
        # Test queries for each intent category (20 per category)
        self.test_queries = {
            "technical": [
                "How do I integrate the API into my application?",
                "Getting 404 error when calling the endpoint",
                "Authentication is not working properly",
                "How to handle rate limiting in my code?",
                "Can you provide API documentation examples?",
                "Getting timeout errors when making requests",
                "How do I set up webhook notifications?",
                "API key validation is failing",
                "Getting 500 server error responses",
                "How to implement retry logic for failed requests?",
                "Authentication token expired, how to refresh?",
                "Getting CORS errors when calling from browser",
                "How to handle pagination in API responses?",
                "API response format is different than expected",
                "Getting permission denied errors",
                "How to debug API integration issues?",
                "SDK installation is failing",
                "Getting connection refused errors",
                "How to implement proper error handling?",
                "API versioning is confusing, need help"
            ],
            "billing": [
                "How much does the premium plan cost?",
                "I want to cancel my subscription",
                "Update my billing information",
                "What's included in the Pro plan?",
                "Refund policy for annual plans",
                "How to upgrade from Basic to Pro?",
                "Billing cycle and payment dates",
                "Can I get a discount for annual billing?",
                "How to change payment method?",
                "What happens if payment fails?",
                "Enterprise plan pricing details",
                "How to downgrade my plan?",
                "Trial period and conversion",
                "Invoice generation and download",
                "Tax calculation and billing",
                "How to add team members to account?",
                "Usage limits and overage charges",
                "Billing support contact information",
                "How to export billing history?",
                "Custom pricing for large organizations"
            ],
            "feature": [
                "Can you add dark mode to the interface?",
                "Need export to PDF functionality",
                "Request for mobile app development",
                "When will you support webhooks?",
                "Add support for Python SDK",
                "Can you implement real-time notifications?",
                "Request for advanced analytics dashboard",
                "Need custom branding options",
                "When will you add multi-language support?",
                "Request for bulk import functionality",
                "Can you add calendar integration?",
                "Need email automation features",
                "Request for API rate limiting controls",
                "When will you support OAuth 2.0?",
                "Need data backup and restore features",
                "Request for custom webhook endpoints",
                "Can you add team collaboration tools?",
                "Need advanced search functionality",
                "Request for automated reporting",
                "When will you add mobile SDK?"
            ]
        }
        
        # Expected intents for each query (for evaluation)
        self.expected_intents = {}
        for intent, queries in self.test_queries.items():
            for query in queries:
                self.expected_intents[query] = intent
    
    def get_all_test_queries(self) -> List[str]:
        """Get all test queries as a flat list"""
        all_queries = []
        for queries in self.test_queries.values():
            all_queries.extend(queries)
        return all_queries
    
    def get_queries_by_intent(self, intent: str) -> List[str]:
        """Get test queries for a specific intent"""
        return self.test_queries.get(intent, [])
    
    def get_expected_intent(self, query: str) -> str:
        """Get expected intent for a query"""
        return self.expected_intents.get(query, "technical")
    
    def get_query_intent_pairs(self) -> List[Tuple[str, str]]:
        """Get all query-intent pairs for evaluation"""
        pairs = []
        for intent, queries in self.test_queries.items():
            for query in queries:
                pairs.append((query, intent))
        return pairs
    
    def get_random_sample(self, size: int = 10) -> List[str]:
        """Get a random sample of test queries"""
        all_queries = self.get_all_test_queries()
        return random.sample(all_queries, min(size, len(all_queries)))
    
    def get_balanced_sample(self, samples_per_intent: int = 5) -> List[str]:
        """Get a balanced sample with equal queries per intent"""
        balanced_sample = []
        for intent in self.test_queries.keys():
            intent_queries = self.test_queries[intent]
            sample_size = min(samples_per_intent, len(intent_queries))
            balanced_sample.extend(random.sample(intent_queries, sample_size))
        return balanced_sample
    
    def get_intent_distribution(self) -> Dict[str, int]:
        """Get distribution of queries across intents"""
        return {intent: len(queries) for intent, queries in self.test_queries.items()}
    
    def validate_query(self, query: str) -> bool:
        """Check if a query exists in test set"""
        return query in self.expected_intents
    
    def add_custom_query(self, query: str, expected_intent: str):
        """Add a custom query to the test set"""
        if expected_intent not in self.test_queries:
            self.test_queries[expected_intent] = []
        
        self.test_queries[expected_intent].append(query)
        self.expected_intents[query] = expected_intent
    
    def get_query_categories(self) -> Dict[str, List[str]]:
        """Get queries organized by subcategories"""
        categories = {
            "technical": {
                "authentication": [
                    "Authentication is not working properly",
                    "API key validation is failing",
                    "Authentication token expired, how to refresh?",
                    "Getting permission denied errors"
                ],
                "errors": [
                    "Getting 404 error when calling the endpoint",
                    "Getting 500 server error responses",
                    "Getting timeout errors when making requests",
                    "Getting CORS errors when calling from browser"
                ],
                "integration": [
                    "How do I integrate the API into my application?",
                    "How to handle rate limiting in my code?",
                    "How do I set up webhook notifications?",
                    "How to implement retry logic for failed requests?"
                ]
            },
            "billing": {
                "pricing": [
                    "How much does the premium plan cost?",
                    "What's included in the Pro plan?",
                    "Enterprise plan pricing details",
                    "Can I get a discount for annual billing?"
                ],
                "subscription": [
                    "I want to cancel my subscription",
                    "How to upgrade from Basic to Pro?",
                    "How to downgrade my plan?",
                    "Trial period and conversion"
                ],
                "payment": [
                    "Update my billing information",
                    "How to change payment method?",
                    "What happens if payment fails?",
                    "Invoice generation and download"
                ]
            },
            "feature": {
                "ui_ux": [
                    "Can you add dark mode to the interface?",
                    "Need custom branding options",
                    "Request for advanced analytics dashboard",
                    "Need advanced search functionality"
                ],
                "functionality": [
                    "Need export to PDF functionality",
                    "Request for bulk import functionality",
                    "Need data backup and restore features",
                    "Request for automated reporting"
                ],
                "integration": [
                    "When will you support webhooks?",
                    "Add support for Python SDK",
                    "Can you implement real-time notifications?",
                    "When will you add mobile SDK?"
                ]
            }
        }
        return categories
    
    def get_difficulty_levels(self) -> Dict[str, List[str]]:
        """Get queries organized by difficulty level"""
        return {
            "easy": [
                "How much does the premium plan cost?",
                "I want to cancel my subscription",
                "Can you add dark mode to the interface?",
                "How do I integrate the API into my application?"
            ],
            "medium": [
                "Getting 404 error when calling the endpoint",
                "How to upgrade from Basic to Pro?",
                "Need export to PDF functionality",
                "Authentication is not working properly"
            ],
            "hard": [
                "How to implement proper error handling?",
                "Custom pricing for large organizations",
                "When will you add mobile SDK?",
                "API versioning is confusing, need help"
            ]
        } 