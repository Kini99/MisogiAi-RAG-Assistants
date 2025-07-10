"""
Load testing for Financial Intelligence RAG System
Tests concurrent request handling and performance metrics
"""

import random
import json
from locust import HttpUser, task, between, events
from typing import Dict, Any


class FinancialRAGUser(HttpUser):
    """Simulates a user interacting with the Financial RAG System"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        # Sample financial queries for testing
        self.sample_queries = [
            "What is the current P/E ratio?",
            "How does the company's revenue compare to last year?",
            "What are the key financial metrics for Q3?",
            "Analyze the debt-to-equity ratio trends",
            "What is the company's market position compared to competitors?",
            "Show me the quarterly earnings growth",
            "What are the main risk factors in the financial statements?",
            "How has the company's cash flow changed over time?",
            "What is the return on equity for the past 3 years?",
            "Analyze the profit margins and their trends"
        ]
        
        # Sample companies for testing
        self.sample_companies = [
            {"id": 1, "name": "Apple Inc."},
            {"id": 2, "name": "Microsoft Corporation"},
            {"id": 3, "name": "Google LLC"},
            {"id": 4, "name": "Amazon.com Inc."},
            {"id": 5, "name": "Tesla Inc."}
        ]
    
    @task(3)
    def submit_financial_query(self):
        """Submit a financial query (most common task)"""
        query = random.choice(self.sample_queries)
        company = random.choice(self.sample_companies)
        
        payload = {
            "query": query,
            "company_id": company["id"],
            "company_name": company["name"]
        }
        
        with self.client.post(
            "/api/v1/query",
            json=payload,
            catch_response=True,
            name="Submit Financial Query"
        ) as response:
            if response.status_code == 200:
                response_data = response.json()
                
                # Validate response structure
                required_fields = ["query_id", "response", "cache_hit", "response_time"]
                if all(field in response_data for field in required_fields):
                    response.success()
                    
                    # Log performance metrics
                    self.log_performance_metrics(
                        "query_submission",
                        response_data["response_time"],
                        response_data["cache_hit"],
                        response_data["tokens_used"]
                    )
                else:
                    response.failure("Invalid response structure")
            elif response.status_code == 429:
                response.success()  # Rate limiting is expected
            else:
                response.failure(f"Unexpected status code: {response.status_code}")
    
    @task(1)
    def get_query_history(self):
        """Get query history"""
        company_id = random.choice(self.sample_companies)["id"]
        
        with self.client.get(
            f"/api/v1/query/history?company_id={company_id}&limit=10",
            catch_response=True,
            name="Get Query History"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get query history: {response.status_code}")
    
    @task(1)
    def list_companies(self):
        """List available companies"""
        with self.client.get(
            "/api/v1/companies",
            catch_response=True,
            name="List Companies"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to list companies: {response.status_code}")
    
    @task(1)
    def get_system_metrics(self):
        """Get system performance metrics"""
        with self.client.get(
            "/api/v1/metrics",
            catch_response=True,
            name="Get System Metrics"
        ) as response:
            if response.status_code == 200:
                response_data = response.json()
                
                # Validate metrics structure
                if "cache" in response_data and "performance" in response_data:
                    response.success()
                else:
                    response.failure("Invalid metrics structure")
            else:
                response.failure(f"Failed to get metrics: {response.status_code}")
    
    @task(1)
    def get_cache_stats(self):
        """Get cache statistics"""
        with self.client.get(
            "/api/v1/cache/stats",
            catch_response=True,
            name="Get Cache Stats"
        ) as response:
            if response.status_code == 200:
                response_data = response.json()
                
                # Validate cache stats
                if "hit_ratio" in response_data:
                    response.success()
                else:
                    response.failure("Invalid cache stats structure")
            else:
                response.failure(f"Failed to get cache stats: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Health check endpoint"""
        with self.client.get(
            "/api/v1/health",
            catch_response=True,
            name="Health Check"
        ) as response:
            if response.status_code == 200:
                response_data = response.json()
                
                if response_data.get("status") in ["healthy", "unhealthy"]:
                    response.success()
                else:
                    response.failure("Invalid health status")
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    def log_performance_metrics(self, operation: str, response_time: float, 
                               cache_hit: bool, tokens_used: int):
        """Log performance metrics for analysis"""
        metrics = {
            "operation": operation,
            "response_time": response_time,
            "cache_hit": cache_hit,
            "tokens_used": tokens_used,
            "timestamp": self.environment.runner.start_time
        }
        
        # In a real scenario, you might send these to a metrics collection service
        print(f"PERFORMANCE_METRIC: {json.dumps(metrics)}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when a test is starting"""
    print("Starting Financial RAG System Load Test")
    print(f"Target: {environment.host}")
    print(f"Users: {environment.runner.user_count}")
    print(f"Spawning rate: {environment.runner.spawn_rate}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when a test is ending"""
    print("Financial RAG System Load Test completed")
    
    # Print summary statistics
    stats = environment.stats
    print("\n=== PERFORMANCE SUMMARY ===")
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Failed requests: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Median response time: {stats.total.median_response_time:.2f}ms")
    print(f"95th percentile: {stats.total.percentile_95:.2f}ms")
    print(f"99th percentile: {stats.total.percentile_99:.2f}ms")
    print(f"Requests per second: {stats.total.current_rps:.2f}")


# Custom event for tracking cache performance
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, start_time, url, **kwargs):
    """Track cache hit rates and other custom metrics"""
    if name == "Submit Financial Query" and response and response.status_code == 200:
        try:
            response_data = response.json()
            if response_data.get("cache_hit"):
                # Log cache hit
                print(f"CACHE_HIT: {name} - Response time: {response_time:.2f}ms")
            else:
                # Log cache miss
                print(f"CACHE_MISS: {name} - Response time: {response_time:.2f}ms")
        except:
            pass 