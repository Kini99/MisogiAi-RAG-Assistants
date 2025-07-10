#!/usr/bin/env python3
"""
Comprehensive test script for Financial Intelligence RAG System
Tests all major components and endpoints
"""

import asyncio
import requests
import json
import time
from typing import Dict, Any


class SystemTester:
    """Comprehensive system tester"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "details": details
        })
    
    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") in ["healthy", "unhealthy"]:
                    self.log_test("Health Check", True, f"Status: {data['status']}")
                    return True
                else:
                    self.log_test("Health Check", False, "Invalid status in response")
                    return False
            else:
                self.log_test("Health Check", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {str(e)}")
            return False
    
    def test_create_company(self) -> int:
        """Test company creation"""
        try:
            company_data = {
                "name": "Test Company Inc.",
                "ticker": "TEST",
                "sector": "Technology",
                "industry": "Software",
                "market_cap": 1000000000.0,
                "revenue": 500000000.0
            }
            
            response = self.session.post(
                f"{self.base_url}/api/v1/companies",
                json=company_data
            )
            
            if response.status_code == 200:
                data = response.json()
                company_id = data.get("id")
                self.log_test("Create Company", True, f"Company ID: {company_id}")
                return company_id
            else:
                self.log_test("Create Company", False, f"Status code: {response.status_code}")
                return None
        except Exception as e:
            self.log_test("Create Company", False, f"Exception: {str(e)}")
            return None
    
    def test_list_companies(self) -> bool:
        """Test listing companies"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/companies")
            if response.status_code == 200:
                companies = response.json()
                self.log_test("List Companies", True, f"Found {len(companies)} companies")
                return True
            else:
                self.log_test("List Companies", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("List Companies", False, f"Exception: {str(e)}")
            return False
    
    def test_submit_query(self, company_id: int = None) -> str:
        """Test query submission"""
        try:
            query_data = {
                "query": "What is the current P/E ratio and how does it compare to industry average?",
                "company_name": "Test Company"
            }
            
            if company_id:
                query_data["company_id"] = company_id
            
            response = self.session.post(
                f"{self.base_url}/api/v1/query",
                json=query_data
            )
            
            if response.status_code == 200:
                data = response.json()
                query_id = data.get("query_id")
                response_time = data.get("response_time", 0)
                cache_hit = data.get("cache_hit", False)
                
                self.log_test(
                    "Submit Query", 
                    True, 
                    f"Query ID: {query_id}, Response time: {response_time:.2f}s, Cache hit: {cache_hit}"
                )
                return query_id
            else:
                self.log_test("Submit Query", False, f"Status code: {response.status_code}")
                return None
        except Exception as e:
            self.log_test("Submit Query", False, f"Exception: {str(e)}")
            return None
    
    def test_get_query_result(self, query_id: str) -> bool:
        """Test getting query result"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/query/{query_id}")
            if response.status_code == 200:
                data = response.json()
                self.log_test("Get Query Result", True, f"Response length: {len(data.get('response', ''))}")
                return True
            else:
                self.log_test("Get Query Result", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Get Query Result", False, f"Exception: {str(e)}")
            return False
    
    def test_get_metrics(self) -> bool:
        """Test getting system metrics"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/metrics")
            if response.status_code == 200:
                data = response.json()
                if "cache" in data and "performance" in data:
                    self.log_test("Get Metrics", True, "Metrics retrieved successfully")
                    return True
                else:
                    self.log_test("Get Metrics", False, "Invalid metrics structure")
                    return False
            else:
                self.log_test("Get Metrics", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Get Metrics", False, f"Exception: {str(e)}")
            return False
    
    def test_cache_stats(self) -> bool:
        """Test cache statistics"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/cache/stats")
            if response.status_code == 200:
                data = response.json()
                hit_ratio = data.get("hit_ratio", 0)
                self.log_test("Cache Stats", True, f"Cache hit ratio: {hit_ratio}%")
                return True
            else:
                self.log_test("Cache Stats", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Cache Stats", False, f"Exception: {str(e)}")
            return False
    
    def test_query_history(self) -> bool:
        """Test query history"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/query/history?limit=5")
            if response.status_code == 200:
                data = response.json()
                query_count = data.get("total_count", 0)
                self.log_test("Query History", True, f"Found {query_count} queries")
                return True
            else:
                self.log_test("Query History", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Query History", False, f"Exception: {str(e)}")
            return False
    
    def test_concurrent_queries(self) -> bool:
        """Test concurrent query handling"""
        try:
            import concurrent.futures
            
            def submit_single_query():
                query_data = {
                    "query": f"Test query {time.time()}",
                    "company_name": "Test Company"
                }
                response = self.session.post(f"{self.base_url}/api/v1/query", json=query_data)
                return response.status_code == 200
            
            # Submit 10 concurrent queries
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(submit_single_query) for _ in range(10)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            success_count = sum(results)
            self.log_test("Concurrent Queries", success_count >= 8, f"{success_count}/10 queries successful")
            return success_count >= 8
        except Exception as e:
            self.log_test("Concurrent Queries", False, f"Exception: {str(e)}")
            return False
    
    def test_performance_targets(self) -> bool:
        """Test performance targets"""
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/v1/query",
                json={"query": "Performance test query", "company_name": "Test Company"}
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                actual_response_time = data.get("response_time", 0)
                
                # Check if response time is under 2 seconds
                if actual_response_time < 2.0:
                    self.log_test(
                        "Performance Target (<2s)", 
                        True, 
                        f"Response time: {actual_response_time:.2f}s"
                    )
                    return True
                else:
                    self.log_test(
                        "Performance Target (<2s)", 
                        False, 
                        f"Response time: {actual_response_time:.2f}s exceeds 2s limit"
                    )
                    return False
            else:
                self.log_test("Performance Target", False, f"Query failed with status {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Performance Target", False, f"Exception: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("🧪 Running Financial Intelligence RAG System Tests")
        print("=" * 60)
        
        # Basic functionality tests
        self.test_health_check()
        company_id = self.test_create_company()
        self.test_list_companies()
        
        # Query processing tests
        query_id = self.test_submit_query(company_id)
        if query_id:
            self.test_get_query_result(query_id)
        
        # System monitoring tests
        self.test_get_metrics()
        self.test_cache_stats()
        self.test_query_history()
        
        # Performance tests
        self.test_concurrent_queries()
        self.test_performance_targets()
        
        # Summary
        print("\n" + "=" * 60)
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        
        print(f"📊 Test Summary:")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests == 0:
            print("\n🎉 All tests passed! System is working correctly.")
        else:
            print(f"\n⚠️  {failed_tests} test(s) failed. Please check the system configuration.")
        
        return failed_tests == 0


def main():
    """Main test function"""
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    tester = SystemTester(base_url)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 