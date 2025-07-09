"""
Real-time monitoring and alerting for RAGAS metrics.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from collections import deque
import threading

from .metrics import MedicalRAGASMetrics
from medical_rag.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGASMonitor:
    """Real-time monitoring for RAGAS metrics with alerting capabilities."""
    
    def __init__(self):
        self.metrics = MedicalRAGASMetrics()
        self.monitoring_data = deque(maxlen=1000)  # Keep last 1000 evaluations
        self.alert_callbacks: List[Callable] = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Alert thresholds
        self.alert_thresholds = {
            "faithfulness": 0.85,  # Alert if below 0.85
            "context_precision": 0.80,  # Alert if below 0.80
            "context_recall": 0.75,  # Alert if below 0.75
            "answer_relevancy": 0.80,  # Alert if below 0.80
            "safety_score": 0.90,  # Alert if below 0.90
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            "response_time_p95": 3.0,  # Alert if p95 > 3 seconds
            "evaluation_time": 10.0,  # Alert if evaluation > 10 seconds
        }
        
        logger.info("Initialized RAGAS Monitor")
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started RAGAS monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Stopped RAGAS monitoring")
    
    def add_evaluation_result(self, result: Dict[str, Any]) -> None:
        """
        Add evaluation result to monitoring data.
        
        Args:
            result: Evaluation result dictionary
        """
        try:
            # Add timestamp if not present
            if "timestamp" not in result:
                result["timestamp"] = datetime.now().isoformat()
            
            # Add to monitoring data
            self.monitoring_data.append(result)
            
            # Check for immediate alerts
            self._check_immediate_alerts(result)
            
        except Exception as e:
            logger.error(f"Error adding evaluation result: {e}")
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add alert callback function.
        
        Args:
            callback: Function to call when alert is triggered
        """
        self.alert_callbacks.append(callback)
        logger.info(f"Added alert callback: {callback.__name__}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current aggregated metrics.
        
        Returns:
            Current metrics summary
        """
        try:
            if not self.monitoring_data:
                return {"status": "no_data", "message": "No monitoring data available"}
            
            # Calculate metrics over recent data (last hour)
            recent_data = self._get_recent_data(hours=1)
            
            if not recent_data:
                return {"status": "no_recent_data", "message": "No recent data available"}
            
            # Aggregate metrics
            aggregated_metrics = self._aggregate_metrics(recent_data)
            
            # Add status information
            status = {
                "monitoring_active": self.monitoring_active,
                "total_evaluations": len(self.monitoring_data),
                "recent_evaluations": len(recent_data),
                "last_evaluation": self.monitoring_data[-1]["timestamp"] if self.monitoring_data else None,
                "alert_thresholds": self.alert_thresholds
            }
            
            return {
                "status": "active",
                "current_metrics": aggregated_metrics,
                "status_info": status
            }
            
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_metrics_history(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get metrics history for specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Metrics history
        """
        try:
            recent_data = self._get_recent_data(hours=hours)
            
            if not recent_data:
                return {"status": "no_data", "message": f"No data available for last {hours} hours"}
            
            # Group by time intervals
            history = self._group_by_time_intervals(recent_data, hours)
            
            return {
                "status": "success",
                "time_period_hours": hours,
                "history": history,
                "summary": self._aggregate_metrics(recent_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get alerts for specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of alerts
        """
        try:
            recent_data = self._get_recent_data(hours=hours)
            alerts = []
            
            for result in recent_data:
                if "quality_check" in result and not result["quality_check"].get("overall_pass", True):
                    alert = {
                        "timestamp": result["timestamp"],
                        "batch_name": result.get("batch_name", "Unknown"),
                        "severity": "high" if "faithfulness" in result["quality_check"].get("failed_metrics", []) else "medium",
                        "failed_metrics": result["quality_check"].get("failed_metrics", []),
                        "warnings": result["quality_check"].get("warnings", [])
                    }
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Check for performance issues
                self._check_performance_alerts()
                
                # Check for trend-based alerts
                self._check_trend_alerts()
                
                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _check_immediate_alerts(self, result: Dict[str, Any]) -> None:
        """Check for immediate alerts in evaluation result."""
        try:
            alerts = []
            
            # Check quality thresholds
            if "quality_check" in result:
                quality_check = result["quality_check"]
                if not quality_check.get("overall_pass", True):
                    failed_metrics = quality_check.get("failed_metrics", [])
                    warnings = quality_check.get("warnings", [])
                    
                    alert = {
                        "type": "quality_threshold",
                        "timestamp": result["timestamp"],
                        "batch_name": result.get("batch_name", "Unknown"),
                        "severity": "critical" if "faithfulness" in failed_metrics else "warning",
                        "failed_metrics": failed_metrics,
                        "warnings": warnings
                    }
                    alerts.append(alert)
            
            # Check performance thresholds
            if "evaluation_time" in result:
                eval_time = result["evaluation_time"]
                if eval_time > self.performance_thresholds["evaluation_time"]:
                    alert = {
                        "type": "performance",
                        "timestamp": result["timestamp"],
                        "batch_name": result.get("batch_name", "Unknown"),
                        "severity": "warning",
                        "message": f"Evaluation time ({eval_time:.2f}s) exceeded threshold"
                    }
                    alerts.append(alert)
            
            # Trigger alert callbacks
            for alert in alerts:
                self._trigger_alerts(alert)
                
        except Exception as e:
            logger.error(f"Error checking immediate alerts: {e}")
    
    def _check_performance_alerts(self) -> None:
        """Check for performance-based alerts."""
        try:
            recent_data = self._get_recent_data(hours=1)
            
            if not recent_data:
                return
            
            # Calculate response time percentiles
            response_times = []
            for result in recent_data:
                if "evaluation_time" in result:
                    response_times.append(result["evaluation_time"])
            
            if response_times:
                response_times.sort()
                p95_index = int(len(response_times) * 0.95)
                p95_time = response_times[p95_index] if p95_index < len(response_times) else response_times[-1]
                
                if p95_time > self.performance_thresholds["response_time_p95"]:
                    alert = {
                        "type": "performance_p95",
                        "timestamp": datetime.now().isoformat(),
                        "severity": "warning",
                        "message": f"P95 response time ({p95_time:.2f}s) exceeded threshold"
                    }
                    self._trigger_alerts(alert)
                    
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    def _check_trend_alerts(self) -> None:
        """Check for trend-based alerts."""
        try:
            recent_data = self._get_recent_data(hours=1)
            
            if len(recent_data) < 5:  # Need at least 5 evaluations for trend analysis
                return
            
            # Calculate trend for key metrics
            metrics_trend = self._calculate_metrics_trend(recent_data)
            
            # Check for declining trends
            for metric, trend in metrics_trend.items():
                if trend < -0.05:  # Declining by more than 5%
                    alert = {
                        "type": "trend_decline",
                        "timestamp": datetime.now().isoformat(),
                        "severity": "warning",
                        "message": f"Declining trend detected for {metric}: {trend:.3f}"
                    }
                    self._trigger_alerts(alert)
                    
        except Exception as e:
            logger.error(f"Error checking trend alerts: {e}")
    
    def _trigger_alerts(self, alert: Dict[str, Any]) -> None:
        """Trigger alert callbacks."""
        try:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback {callback.__name__}: {e}")
                    
        except Exception as e:
            logger.error(f"Error triggering alerts: {e}")
    
    def _get_recent_data(self, hours: int) -> List[Dict[str, Any]]:
        """Get recent data within specified hours."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_data = []
            
            for result in self.monitoring_data:
                try:
                    result_time = datetime.fromisoformat(result["timestamp"])
                    if result_time >= cutoff_time:
                        recent_data.append(result)
                except (ValueError, KeyError):
                    continue
            
            return recent_data
            
        except Exception as e:
            logger.error(f"Error getting recent data: {e}")
            return []
    
    def _aggregate_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics from evaluation data."""
        try:
            aggregated = {}
            metric_lists = {}
            
            # Collect all metric values
            for result in data:
                if "metrics" in result:
                    for metric, value in result["metrics"].items():
                        if isinstance(value, (int, float)):
                            if metric not in metric_lists:
                                metric_lists[metric] = []
                            metric_lists[metric].append(value)
            
            # Calculate averages
            for metric, values in metric_lists.items():
                if values:
                    aggregated[metric] = sum(values) / len(values)
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating metrics: {e}")
            return {}
    
    def _group_by_time_intervals(self, data: List[Dict[str, Any]], hours: int) -> Dict[str, Any]:
        """Group data by time intervals."""
        try:
            intervals = {}
            interval_hours = max(1, hours // 24)  # Group by day for 24h+, by hour otherwise
            
            for result in data:
                try:
                    result_time = datetime.fromisoformat(result["timestamp"])
                    interval_key = result_time.strftime(f"%Y-%m-%d %H:{interval_hours:02d}:00")
                    
                    if interval_key not in intervals:
                        intervals[interval_key] = []
                    intervals[interval_key].append(result)
                    
                except (ValueError, KeyError):
                    continue
            
            # Aggregate each interval
            history = {}
            for interval, interval_data in intervals.items():
                history[interval] = self._aggregate_metrics(interval_data)
            
            return history
            
        except Exception as e:
            logger.error(f"Error grouping by time intervals: {e}")
            return {}
    
    def _calculate_metrics_trend(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate trend for metrics over time."""
        try:
            trends = {}
            
            # Sort data by timestamp
            sorted_data = sorted(data, key=lambda x: x.get("timestamp", ""))
            
            if len(sorted_data) < 2:
                return trends
            
            # Calculate trend for each metric
            for result in sorted_data:
                if "metrics" in result:
                    for metric, value in result["metrics"].items():
                        if isinstance(value, (int, float)):
                            if metric not in trends:
                                trends[metric] = []
                            trends[metric].append(value)
            
            # Calculate linear trend (simple slope)
            for metric, values in trends.items():
                if len(values) >= 2:
                    # Simple linear trend calculation
                    n = len(values)
                    x = list(range(n))
                    y = values
                    
                    # Calculate slope
                    sum_x = sum(x)
                    sum_y = sum(y)
                    sum_xy = sum(x[i] * y[i] for i in range(n))
                    sum_x2 = sum(x[i] ** 2 for i in range(n))
                    
                    if n * sum_x2 - sum_x ** 2 != 0:
                        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                        trends[metric] = slope
                    else:
                        trends[metric] = 0.0
                else:
                    trends[metric] = 0.0
            
            return trends
            
        except Exception as e:
            logger.error(f"Error calculating metrics trend: {e}")
            return {} 