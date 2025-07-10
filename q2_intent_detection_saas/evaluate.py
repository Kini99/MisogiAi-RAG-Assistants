#!/usr/bin/env python3
"""
Main Evaluation Script for Customer Support System
Runs comprehensive evaluation and generates reports
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from evaluation.evaluator import Evaluator
from support_system import CustomerSupportSystem

def main():
    parser = argparse.ArgumentParser(description='Evaluate Customer Support System')
    parser.add_argument('--mode', choices=['full', 'intent', 'balanced', 'health'], 
                       default='full', help='Evaluation mode')
    parser.add_argument('--intent', choices=['technical', 'billing', 'feature'],
                       help='Specific intent to evaluate (for intent mode)')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of samples per intent (for balanced mode)')
    parser.add_argument('--output-dir', default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick evaluation with fewer queries')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CUSTOMER SUPPORT SYSTEM EVALUATION")
    print("="*60)
    
    # Check system health first
    print("\nChecking system health...")
    support_system = CustomerSupportSystem()
    health = support_system.health_check()
    
    print(f"System Status: {health['system_status']}")
    print(f"LLM Services: {health['llm_services']}")
    print(f"Processors Available: {health['processors_available']}")
    
    if health['system_status'] == 'degraded':
        print("\n⚠️  WARNING: System is in degraded state. Evaluation may not be optimal.")
        response = input("Continue with evaluation? (y/N): ")
        if response.lower() != 'y':
            print("Evaluation cancelled.")
            return
    
    # Initialize evaluator
    evaluator = Evaluator(output_dir=args.output_dir)
    
    try:
        if args.mode == 'health':
            print("\n" + "="*40)
            print("SYSTEM HEALTH CHECK")
            print("="*40)
            print(f"Status: {health['system_status']}")
            print(f"Average Response Time: {health['avg_response_time']:.3f}s")
            print(f"Total Queries Processed: {health['total_queries_processed']}")
            print(f"LLM Services: {health['llm_services']}")
            return
        
        elif args.mode == 'intent':
            if not args.intent:
                print("Error: --intent is required for intent evaluation mode")
                return
            
            print(f"\nEvaluating {args.intent} intent...")
            results = evaluator.run_intent_evaluation(args.intent)
            
        elif args.mode == 'balanced':
            print(f"\nRunning balanced evaluation with {args.samples} samples per intent...")
            results = evaluator.run_balanced_evaluation(args.samples)
            
        else:  # full mode
            if args.quick:
                print("\nRunning quick evaluation...")
                results = evaluator.run_balanced_evaluation(3)  # 3 samples per intent
            else:
                print("\nRunning full evaluation with all test queries...")
                results = evaluator.run_full_evaluation()
        
        # Print summary
        evaluator.print_summary()
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 