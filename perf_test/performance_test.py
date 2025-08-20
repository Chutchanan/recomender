# perf_test/performance_test.py - Performance testing for Restaurant Recommendation API
import pandas as pd
import requests
import time
import numpy as np
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import statistics
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configuration
API_BASE_URL = "http://localhost:8000"
TARGET_RPS = 30
TEST_DURATION = 60  # seconds
DATA_FILE = Path("../data/requests.parquet")

class PerformanceTest:
    def __init__(self):
        self.results = []
        self.errors = []
        self.start_time = None
        
    def load_test_data(self):
        """Load requests from parquet file or generate synthetic data"""
        print("📊 Loading test data...")
        
        if DATA_FILE.exists():
            # Use existing requests file - COPIED FROM ORIGINAL CODE
            df = pd.read_parquet(DATA_FILE)
            print(f"   Loaded {len(df):,} test requests from file")
            
            # Convert to test cases - EXACT COPY OF ORIGINAL APPROACH WITH LIMITS
            test_requests = []
            for _, row in df.iterrows():
                # Convert numpy array to Python list for JSON serialization
                candidate_ids = row["candidate_restaurant_ids"]
                if isinstance(candidate_ids, np.ndarray):
                    candidate_ids = candidate_ids.tolist()
                
                # LIMIT candidate restaurants to 500 (API validation limit)
                if len(candidate_ids) > 500:
                    candidate_ids = candidate_ids[:500]
                
                test_requests.append({
                    "user_id": str(row["user_id"]),  # Ensure string for our API
                    "payload": {
                        "candidate_restaurant_ids": candidate_ids,
                        "latitude": float(row["latitude"]),  # Ensure float
                        "longitude": float(row["longitude"]),  # Ensure float
                        "size": min(int(row["size"]), 100),  # Cap at 100 for our API validation
                        "max_dist": max(min(int(row["max_dist"]), 50000), 1000),  # Cap and ensure minimum
                        "sort_dist": bool(row["sort_dist"])  # Ensure bool
                    }
                })
            
        else:
            # Generate synthetic test data
            print("   Generating synthetic test data...")
            df = self.generate_synthetic_requests()
            
            # Convert synthetic data to test cases
            test_requests = []
            for _, row in df.iterrows():
                test_requests.append({
                    "user_id": str(row["user_id"]),
                    "payload": {
                        "candidate_restaurant_ids": row["candidate_restaurant_ids"],
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "size": int(row["size"]),
                        "max_dist": int(row["max_dist"]),
                        "sort_dist": bool(row["sort_dist"])
                    }
                })
        
        return test_requests
    
    def generate_synthetic_requests(self, count=2000):
        """Generate synthetic test requests"""
        # Bangkok area coordinates
        lat_range = (13.5, 14.0)
        lon_range = (100.3, 100.9)
        
        # Generate user IDs from our known range (0-999999)
        user_ids = np.random.randint(0, 1000, size=count)
        
        # Generate restaurant candidate lists (use first 100 restaurants which exist)
        restaurant_pools = []
        for _ in range(count):
            # Random number of candidates (5-15) from restaurants that actually exist
            num_candidates = np.random.randint(5, 16)
            candidates = np.random.randint(0, 100, size=num_candidates).tolist()  # 0-99 range
            restaurant_pools.append(candidates)
        
        data = {
            "user_id": user_ids,
            "candidate_restaurant_ids": restaurant_pools,
            "latitude": np.random.uniform(lat_range[0], lat_range[1], count),
            "longitude": np.random.uniform(lon_range[0], lon_range[1], count),
            "size": np.random.randint(5, 21, count),  # 5-20 recommendations
            "max_dist": np.random.choice([20000, 30000, 50000], count, p=[0.3, 0.4, 0.3]),  # Larger distances
            "sort_dist": np.random.choice([True, False], count, p=[0.2, 0.8])  # 20% sort by distance
        }
        
        return pd.DataFrame(data)
    
    def send_request(self, session, user_id, payload):
        """Send single API request - COPIED FROM ORIGINAL"""
        url = f"{API_BASE_URL}/recommend/{user_id}"
        
        try:
            start_time = time.time()
            response = session.post(url, json=payload, timeout=10)  # Increased timeout
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # ms
            
            result = {
                "timestamp": start_time,
                "response_time": response_time,
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "user_id": user_id
            }
            
            if response.status_code == 200:
                data = response.json()
                result["restaurants_count"] = len(data.get("restaurants", []))
                result["processing_time_ms"] = data.get("processing_time_ms", 0)
            else:
                # Log error responses for debugging - ENHANCED
                result["error_detail"] = response.text[:500]
                result["error_type"] = f"HTTP_{response.status_code}"
                
                # Log first few failures for debugging
                if len([r for r in self.results if not r.get("success", True)]) < 5:
                    print(f"   DEBUG - Request failed:")
                    print(f"      User ID: {user_id}")
                    print(f"      Status: {response.status_code}")
                    print(f"      Payload: {payload}")
                    print(f"      Response: {response.text[:200]}")
            
            return result
            
        except requests.exceptions.Timeout:
            return {
                "timestamp": time.time(),
                "response_time": 10000,  # Timeout
                "status_code": 0,
                "success": False,
                "error": "Request timeout",
                "error_type": "timeout",
                "user_id": user_id
            }
        except requests.exceptions.ConnectionError:
            return {
                "timestamp": time.time(),
                "response_time": 0,
                "status_code": 0,
                "success": False,
                "error": "Connection error",
                "error_type": "connection",
                "user_id": user_id
            }
        except Exception as e:
            return {
                "timestamp": time.time(),
                "response_time": 0,
                "status_code": 0,
                "success": False,
                "error": str(e),
                "error_type": "exception",
                "user_id": user_id
            }
    
    def test_api_health(self):
        """Test API health and basic functionality"""
        print("🔍 Testing API health...")
        session = requests.Session()
        
        try:
            # Health check
            health_response = session.get(f"{API_BASE_URL}/health", timeout=5)
            if health_response.status_code != 200:
                raise Exception(f"Health check failed: {health_response.status_code}")
            
            health_data = health_response.json()
            print(f"   ✅ Health: {health_data.get('status', 'unknown')}")
            print(f"   ✅ Model loaded: {health_data.get('model_loaded', False)}")
            print(f"   ✅ Database connected: {health_data.get('database_connected', False)}")
            
            # Test single recommendation with realistic parameters
            test_payload = {
                "candidate_restaurant_ids": list(range(10)),  # Use first 10 restaurants
                "latitude": 13.7563,
                "longitude": 100.5018,
                "size": 5,
                "max_dist": 50000,  # 50km radius
                "sort_dist": False
            }
            
            test_response = session.post(f"{API_BASE_URL}/recommend/0", json=test_payload, timeout=10)
            if test_response.status_code == 200:
                test_data = test_response.json()
                print(f"   ✅ Test recommendation: {len(test_data.get('restaurants', []))} restaurants")
                print(f"   ✅ Processing time: {test_data.get('processing_time_ms', 0):.1f}ms")
            else:
                print(f"   ⚠️  Test recommendation failed: {test_response.status_code}")
                print(f"      Response: {test_response.text[:100]}")
            
            return True
            
        except Exception as e:
            print(f"❌ API health check failed: {e}")
            return False
        finally:
            session.close()
    
    def run_performance_test(self):
        """Run the main performance test"""
        print("🚀 Starting Performance Test")
        print(f"   Target: {TARGET_RPS} RPS for {TEST_DURATION} seconds")
        print(f"   Total requests: {TARGET_RPS * TEST_DURATION:,}")
        print()
        
        # Test API health first
        if not self.test_api_health():
            return
        
        print()
        
        # Load test data
        test_requests = self.load_test_data()
        
        # Prepare requests (cycle through if needed)
        total_requests_needed = TARGET_RPS * TEST_DURATION
        if len(test_requests) < total_requests_needed:
            # Repeat requests if we don't have enough
            multiplier = (total_requests_needed // len(test_requests)) + 1
            test_requests = (test_requests * multiplier)[:total_requests_needed]
        else:
            test_requests = test_requests[:total_requests_needed]
        
        print(f"📋 Prepared {len(test_requests):,} test requests")
        print()
        print("⏱️  Starting load test...")
        
        # Execute load test
        self.start_time = time.time()
        
        # Use moderate concurrency to avoid overwhelming the API
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for i, request in enumerate(test_requests):
                # Calculate when this request should be sent
                target_time = self.start_time + (i / TARGET_RPS)
                
                # Wait until it's time to send this request
                current_time = time.time()
                if current_time < target_time:
                    time.sleep(target_time - current_time)
                
                # Submit request
                session = requests.Session()  # Create session per request to avoid conflicts
                future = executor.submit(
                    self.send_request, 
                    session, 
                    request["user_id"], 
                    request["payload"]
                )
                futures.append((future, session))
                
                # Small delay to prevent connection flooding
                time.sleep(0.001)  # 1ms delay
                
                # Progress reporting
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - self.start_time
                    current_rps = (i + 1) / elapsed
                    print(f"   Sent {i + 1:,} requests ({elapsed:.1f}s elapsed, {current_rps:.1f} RPS)")
            
            # Collect all results
            print("📊 Collecting results...")
            for future, session in futures:
                try:
                    result = future.result(timeout=15)
                    self.results.append(result)
                except Exception as e:
                    self.errors.append(str(e))
                finally:
                    session.close()
        
        # Analyze results
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze and report test results"""
        print("\n" + "="*60)
        print("📈 PERFORMANCE TEST RESULTS")
        print("="*60)
        
        if not self.results:
            print("❌ No results collected")
            return
        
        # Basic metrics
        total_requests = len(self.results)
        successful_requests = len([r for r in self.results if r["success"]])
        failed_requests = total_requests - successful_requests
        
        success_rate = (successful_requests / total_requests) * 100
        
        print(f"📊 Request Summary:")
        print(f"   Total Requests: {total_requests:,}")
        print(f"   Successful: {successful_requests:,} ({success_rate:.1f}%)")
        print(f"   Failed: {failed_requests:,}")
        
        if self.errors:
            print(f"   Errors: {len(self.errors)}")
        
        # Response time analysis
        successful_results = [r for r in self.results if r["success"]]
        if successful_results:
            response_times = [r["response_time"] for r in successful_results]
            
            print(f"\n⏱️  Response Time Analysis:")
            print(f"   Mean: {statistics.mean(response_times):.1f}ms")
            print(f"   Median: {statistics.median(response_times):.1f}ms")
            print(f"   95th Percentile: {np.percentile(response_times, 95):.1f}ms")
            print(f"   99th Percentile: {np.percentile(response_times, 99):.1f}ms")
            print(f"   Max: {max(response_times):.1f}ms")
            print(f"   Min: {min(response_times):.1f}ms")
            
            # Processing time analysis (internal API time)
            processing_times = [r.get("processing_time_ms", 0) for r in successful_results if "processing_time_ms" in r]
            if processing_times:
                print(f"\n🔧 Internal Processing Time:")
                print(f"   Mean: {statistics.mean(processing_times):.1f}ms")
                print(f"   95th Percentile: {np.percentile(processing_times, 95):.1f}ms")
        
        # Throughput analysis
        if self.start_time and self.results:
            test_duration = max(r["timestamp"] for r in self.results) - self.start_time
            actual_rps = total_requests / test_duration
            
            print(f"\n🚀 Throughput Analysis:")
            print(f"   Test Duration: {test_duration:.1f}s")
            print(f"   Actual RPS: {actual_rps:.1f}")
            print(f"   Target RPS: {TARGET_RPS}")
        
        # Requirements check
        print(f"\n✅ Requirements Check:")
        
        # Check 95th percentile requirement
        if successful_results:
            p95_response_time = np.percentile([r["response_time"] for r in successful_results], 95)
            p95_requirement = p95_response_time <= 100
            print(f"   95th percentile ≤ 100ms: {'✅ PASS' if p95_requirement else '❌ FAIL'} ({p95_response_time:.1f}ms)")
        
        # Check success rate
        success_requirement = success_rate >= 99
        print(f"   Success rate ≥ 99%: {'✅ PASS' if success_requirement else '❌ FAIL'} ({success_rate:.1f}%)")
        
        # Check throughput
        if 'actual_rps' in locals():
            throughput_requirement = actual_rps >= 30
            print(f"   Throughput ≥ 30 RPS: {'✅ PASS' if throughput_requirement else '❌ FAIL'} ({actual_rps:.1f} RPS)")
        
        # Restaurant count analysis - COPIED FROM ORIGINAL
        restaurant_counts = [r.get("restaurants_count", 0) for r in successful_results if "restaurants_count" in r]
        if restaurant_counts:
            print(f"\n🍽️  Restaurant Results:")
            print(f"   Avg restaurants returned: {statistics.mean(restaurant_counts):.1f}")
            print(f"   Max restaurants returned: {max(restaurant_counts)}")
            print(f"   Min restaurants returned: {min(restaurant_counts)}")
            
            # Additional analysis for our debugging
            requests_with_restaurants = len([c for c in restaurant_counts if c > 0])
            print(f"   Requests with restaurants: {requests_with_restaurants}/{len(restaurant_counts)} ({requests_with_restaurants/len(restaurant_counts)*100:.1f}%)")
            
            zero_count = restaurant_counts.count(0)
            if zero_count > 0:
                print(f"   Zero restaurants: {zero_count} requests")
            if zero_count < len(restaurant_counts):
                non_zero = [c for c in restaurant_counts if c > 0]
                print(f"   Non-zero average: {statistics.mean(non_zero):.1f}")
        else:
            print(f"\n🍽️  Restaurant Results: No data available")
        if failed_requests > 0:
            print(f"\n❌ Error Analysis:")
            error_types = {}
            status_codes = {}
            sample_errors = []
            
            for result in self.results:
                if not result["success"]:
                    # Count error types
                    error_type = result.get("error_type", "unknown")
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                    # Count status codes
                    status_code = result.get("status_code", 0)
                    status_codes[status_code] = status_codes.get(status_code, 0) + 1
                    
                    # Collect sample errors
                    if len(sample_errors) < 3:
                        sample_errors.append(result)
            
            print(f"   Error types:")
            for error_type, count in error_types.items():
                print(f"      {error_type}: {count} occurrences")
            
            print(f"   Status codes:")
            for status_code, count in status_codes.items():
                print(f"      {status_code}: {count} occurrences")
            
            print(f"   Sample errors:")
            for i, error in enumerate(sample_errors):
                print(f"      Error {i+1}:")
                print(f"         User ID: {error.get('user_id', 'unknown')}")
                print(f"         Status: {error.get('status_code', 'unknown')}")
                print(f"         Error: {error.get('error_detail', error.get('error', 'unknown'))[:100]}")
                
            # Specific recommendations based on errors
            if any("404" in str(code) for code in status_codes.keys()):
                print(f"\n💡 Recommendation: Check if user IDs exist in database")
            if any("422" in str(code) for code in status_codes.keys()):
                print(f"💡 Recommendation: Check request validation (size, coordinates, etc.)")
            if any("500" in str(code) for code in status_codes.keys()):
                print(f"💡 Recommendation: Check API logs - internal server error")

def main():
    """Main function"""
    print("🏁 Restaurant Recommendation API Performance Test")
    print("=" * 60)
    
    test = PerformanceTest()
    
    try:
        test.run_performance_test()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()