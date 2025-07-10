import json
import hashlib
import time
from typing import Optional, Any, Dict, List
import redis
from redis.connection import ConnectionPool
from app.core.config import settings
import structlog

logger = structlog.get_logger()


class RedisClient:
    """Redis client with connection pooling and caching strategies"""
    
    def __init__(self):
        self.pool = ConnectionPool.from_url(
            settings.redis_url,
            max_connections=50,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={},
            decode_responses=True
        )
        self.client = redis.Redis(connection_pool=self.pool)
        
    def _generate_cache_key(self, query: str, company_id: Optional[int] = None) -> str:
        """Generate cache key from query and company"""
        key_parts = [query]
        if company_id:
            key_parts.append(str(company_id))
        key_string = "|".join(key_parts)
        return f"financial_rag:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _get_ttl(self, query_type: str) -> int:
        """Get TTL based on query type"""
        if query_type == "realtime":
            return settings.cache_ttl_realtime
        elif query_type == "historical":
            return settings.cache_ttl_historical
        elif query_type == "popular":
            return settings.cache_ttl_popular
        else:
            return settings.cache_ttl_realtime
    
    async def get(self, query: str, company_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        try:
            cache_key = self._generate_cache_key(query, company_id)
            cached_data = self.client.get(cache_key)
            
            if cached_data:
                # Update hit count
                self.client.hincrby(f"{cache_key}:stats", "hits", 1)
                self.client.hset(f"{cache_key}:stats", "last_accessed", time.time())
                
                logger.info("Cache hit", cache_key=cache_key)
                return json.loads(cached_data)
            
            logger.info("Cache miss", cache_key=cache_key)
            return None
            
        except Exception as e:
            logger.error("Redis get error", error=str(e))
            return None
    
    async def set(self, query: str, response: Dict[str, Any], 
                  company_id: Optional[int] = None, query_type: str = "realtime") -> bool:
        """Set cached response"""
        try:
            cache_key = self._generate_cache_key(query, company_id)
            ttl = self._get_ttl(query_type)
            
            # Store response
            self.client.setex(cache_key, ttl, json.dumps(response))
            
            # Store metadata
            metadata = {
                "query": query,
                "company_id": company_id,
                "query_type": query_type,
                "created_at": time.time(),
                "ttl": ttl
            }
            self.client.hmset(f"{cache_key}:metadata", metadata)
            
            # Initialize stats
            self.client.hmset(f"{cache_key}:stats", {
                "hits": 0,
                "created_at": time.time(),
                "last_accessed": time.time()
            })
            
            logger.info("Cache set", cache_key=cache_key, ttl=ttl)
            return True
            
        except Exception as e:
            logger.error("Redis set error", error=str(e))
            return False
    
    async def delete(self, query: str, company_id: Optional[int] = None) -> bool:
        """Delete cached response"""
        try:
            cache_key = self._generate_cache_key(query, company_id)
            self.client.delete(cache_key, f"{cache_key}:metadata", f"{cache_key}:stats")
            logger.info("Cache deleted", cache_key=cache_key)
            return True
        except Exception as e:
            logger.error("Redis delete error", error=str(e))
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            keys = self.client.keys("financial_rag:*")
            total_keys = len(keys)
            
            # Calculate hit ratio
            total_hits = 0
            total_requests = 0
            
            for key in keys:
                if key.endswith(":stats"):
                    stats = self.client.hgetall(key)
                    hits = int(stats.get("hits", 0))
                    total_hits += hits
                    total_requests += hits + 1  # +1 for the initial miss
            
            hit_ratio = (total_hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "total_keys": total_keys,
                "total_hits": total_hits,
                "total_requests": total_requests,
                "hit_ratio": round(hit_ratio, 2),
                "memory_usage": self.client.info()["used_memory_human"]
            }
            
        except Exception as e:
            logger.error("Redis stats error", error=str(e))
            return {}
    
    async def clear_expired(self) -> int:
        """Clear expired cache entries"""
        try:
            # Redis automatically expires keys, but we can clean up metadata
            keys = self.client.keys("financial_rag:*:metadata")
            cleared = 0
            
            for key in keys:
                cache_key = key.replace(":metadata", "")
                if not self.client.exists(cache_key):
                    self.client.delete(key, key.replace(":metadata", ":stats"))
                    cleared += 1
            
            logger.info("Cleared expired cache entries", count=cleared)
            return cleared
            
        except Exception as e:
            logger.error("Redis clear expired error", error=str(e))
            return 0
    
    async def get_popular_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular cached queries"""
        try:
            keys = self.client.keys("financial_rag:*:stats")
            popular_queries = []
            
            for key in keys:
                stats = self.client.hgetall(key)
                hits = int(stats.get("hits", 0))
                if hits > 0:
                    cache_key = key.replace(":stats", "")
                    metadata_key = key.replace(":stats", ":metadata")
                    metadata = self.client.hgetall(metadata_key)
                    
                    popular_queries.append({
                        "query": metadata.get("query", ""),
                        "hits": hits,
                        "last_accessed": stats.get("last_accessed", 0)
                    })
            
            # Sort by hits and return top results
            popular_queries.sort(key=lambda x: x["hits"], reverse=True)
            return popular_queries[:limit]
            
        except Exception as e:
            logger.error("Redis popular queries error", error=str(e))
            return []


# Global Redis client instance
redis_client = RedisClient() 