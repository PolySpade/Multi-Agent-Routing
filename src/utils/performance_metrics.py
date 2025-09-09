import time
import json
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict
import os

@dataclass
class RouteMetrics:
    request_id: str
    computation_time: float
    route_distance: float
    route_safety_score: float
    success: bool
    timestamp: float
    algorithm_used: str = 'astar'

class PerformanceMonitor:
    """Performance monitoring for the MAS-FRO system"""
    
    def __init__(self):
        self.metrics: List[RouteMetrics] = []
        self.system_start_time = time.time()
    
    def record_route_computation(self, metrics: RouteMetrics):
        """Record route computation metrics"""
        self.metrics.append(metrics)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive performance statistics"""
        if not self.metrics:
            return {
                'total_requests': 0,
                'successful_routes': 0,
                'success_rate': 0.0,
                'avg_computation_time': 0.0,
                'avg_route_distance': 0.0,
                'avg_safety_score': 0.0
            }
        
        successful_routes = [m for m in self.metrics if m.success]
        failed_routes = [m for m in self.metrics if not m.success]
        
        # Calculate percentiles for computation time
        if successful_routes:
            computation_times = [m.computation_time for m in successful_routes]
            
            stats = {
                'total_requests': len(self.metrics),
                'successful_routes': len(successful_routes),
                'failed_routes': len(failed_routes),
                'success_rate': len(successful_routes) / len(self.metrics),
                'system_uptime': time.time() - self.system_start_time,
                'avg_computation_time': statistics.mean(computation_times),
                'median_computation_time': statistics.median(computation_times),
                'p95_computation_time': sorted(computation_times)[int(0.95 * len(computation_times))] if computation_times else 0,
                'max_computation_time': max(computation_times) if computation_times else 0,
                'avg_route_distance': statistics.mean([m.route_distance for m in successful_routes]),
                'avg_safety_score': statistics.mean([m.route_safety_score for m in successful_routes])
            }
        else:
            stats = {
                'total_requests': len(self.metrics),
                'successful_routes': 0,
                'failed_routes': len(failed_routes),
                'success_rate': 0.0,
                'system_uptime': time.time() - self.system_start_time,
                'avg_computation_time': 0.0,
                'median_computation_time': 0.0,
                'p95_computation_time': 0.0,
                'max_computation_time': 0.0,
                'avg_route_distance': 0.0,
                'avg_safety_score': 0.0
            }
        
        return stats
    
    def save_metrics(self, filepath: str):
        """Save metrics to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump([asdict(m) for m in self.metrics], f, indent=2)
    
    def generate_report(self) -> str:
        """Generate a text report of performance metrics"""
        stats = self.get_statistics()
        
        report = "="*50 + "\n"
        report += "MAS-FRO PERFORMANCE REPORT\n"
        report += "="*50 + "\n\n"
        
        report += f"Total Requests: {stats['total_requests']}\n"
        report += f"Successful Routes: {stats['successful_routes']}\n"
        report += f"Success Rate: {stats['success_rate']:.2%}\n"
        report += f"System Uptime: {stats['system_uptime']:.1f} seconds\n\n"
        
        if stats['successful_routes'] > 0:
            report += "TIMING METRICS:\n"
            report += f"  Average Computation Time: {stats['avg_computation_time']:.3f}s\n"
            report += f"  Median Computation Time: {stats['median_computation_time']:.3f}s\n"
            report += f"  95th Percentile: {stats['p95_computation_time']:.3f}s\n"
            report += f"  Maximum Time: {stats['max_computation_time']:.3f}s\n\n"
            
            report += "ROUTE QUALITY METRICS:\n"
            report += f"  Average Route Distance: {stats['avg_route_distance']:.0f}m\n"
            report += f"  Average Safety Score: {stats['avg_safety_score']:.3f}\n"
        
        return report

class EvaluationFramework:
    """Framework for comparing MAS-FRO against baseline algorithms"""
    
    def __init__(self):
        self.baseline_monitor = PerformanceMonitor()
        self.mas_fro_monitor = PerformanceMonitor()
    
    def record_baseline_result(self, metrics: RouteMetrics):
        """Record baseline algorithm result"""
        metrics.algorithm_used = 'baseline'
        self.baseline_monitor.record_route_computation(metrics)
    
    def record_mas_fro_result(self, metrics: RouteMetrics):
        """Record MAS-FRO algorithm result"""
        metrics.algorithm_used = 'mas_fro'
        self.mas_fro_monitor.record_route_computation(metrics)
    
    def generate_comparison_report(self) -> str:
        """Generate comparison report between algorithms"""
        baseline_stats = self.baseline_monitor.get_statistics()
        mas_fro_stats = self.mas_fro_monitor.get_statistics()
        
        report = "="*60 + "\n"
        report += "MAS-FRO vs BASELINE COMPARISON REPORT\n"
        report += "="*60 + "\n\n"
        
        # Success Rate Comparison
        report += "SUCCESS RATES:\n"
        report += f"  Baseline: {baseline_stats['success_rate']:.2%}\n"
        report += f"  MAS-FRO:  {mas_fro_stats['success_rate']:.2%}\n"
        if baseline_stats['success_rate'] > 0:
            improvement = (mas_fro_stats['success_rate'] - baseline_stats['success_rate']) / baseline_stats['success_rate'] * 100
            report += f"  Improvement: {improvement:+.1f}%\n\n"
        
        # Safety Score Comparison
        if baseline_stats['avg_safety_score'] > 0 and mas_fro_stats['avg_safety_score'] > 0:
            report += "SAFETY SCORES:\n"
            report += f"  Baseline: {baseline_stats['avg_safety_score']:.3f}\n"
            report += f"  MAS-FRO:  {mas_fro_stats['avg_safety_score']:.3f}\n"
            safety_improvement = (mas_fro_stats['avg_safety_score'] - baseline_stats['avg_safety_score']) / baseline_stats['avg_safety_score'] * 100
            report += f"  Improvement: {safety_improvement:+.1f}%\n\n"
        
        # Computation Time Comparison
        if baseline_stats['avg_computation_time'] > 0 and mas_fro_stats['avg_computation_time'] > 0:
            report += "COMPUTATION TIME:\n"
            report += f"  Baseline: {baseline_stats['avg_computation_time']:.3f}s\n"
            report += f"  MAS-FRO:  {mas_fro_stats['avg_computation_time']:.3f}s\n"
            time_difference = (mas_fro_stats['avg_computation_time'] - baseline_stats['avg_computation_time']) / baseline_stats['avg_computation_time'] * 100
            report += f"  Difference: {time_difference:+.1f}%\n\n"
        
        # Distance Comparison
        if baseline_stats['avg_route_distance'] > 0 and mas_fro_stats['avg_route_distance'] > 0:
            report += "DISTANCE COMPARISON:\n"
            report += f"  Baseline: {baseline_stats['avg_route_distance']:.0f}m\n"
            report += f"  MAS-FRO:  {mas_fro_stats['avg_route_distance']:.0f}m\n"
            distance_difference = (mas_fro_stats['avg_route_distance'] - baseline_stats['avg_route_distance']) / baseline_stats['avg_route_distance'] * 100
            report += f"  Trade-off: {distance_difference:+.1f}%\n"
        
        return report
    
    def save_comparison_results(self, filepath: str):
        """Save detailed comparison results to file"""
        results = {
            'baseline_stats': self.baseline_monitor.get_statistics(),
            'mas_fro_stats': self.mas_fro_monitor.get_statistics(),
            'baseline_metrics': [asdict(m) for m in self.baseline_monitor.metrics],
            'mas_fro_metrics': [asdict(m) for m in self.mas_fro_monitor.metrics],
            'comparison_report': self.generate_comparison_report(),
            'timestamp': time.time()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)