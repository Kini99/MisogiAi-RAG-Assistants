#!/usr/bin/env python3
"""
Medical Knowledge Assistant RAG System Demo

This demo showcases the complete medical RAG pipeline including:
1. Document processing and vector storage
2. Medical query processing
3. RAGAS evaluation
4. Real-time monitoring
5. Safety validation
"""

import os
import time
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up environment variables for demo."""
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not set. Some features may not work.")
        print("   Set your OpenAI API key: export OPENAI_API_KEY='your_key_here'")
        return False
    return True

def create_sample_medical_documents():
    """Create sample medical documents for the demo."""
    documents = {
        "diabetes_guidelines.txt": """
Diabetes Management Guidelines

Diabetes is a chronic condition that requires careful management. Key components include:

1. Blood Sugar Monitoring
- Regular monitoring of blood glucose levels
- Target ranges: 80-130 mg/dL before meals, <180 mg/dL after meals
- Use of continuous glucose monitors for better control

2. Diet Management
- Carbohydrate counting and meal planning
- Low glycemic index foods
- Regular meal timing
- Consultation with registered dietitians

3. Exercise
- Regular physical activity (150 minutes/week)
- Both aerobic and strength training
- Blood sugar monitoring before and after exercise
- Adjustments for exercise-related hypoglycemia

4. Medication
- Insulin therapy for Type 1 diabetes
- Oral medications and/or insulin for Type 2 diabetes
- Regular medication review and adjustment
- Proper injection techniques and site rotation

5. Complications Prevention
- Regular eye examinations
- Foot care and examination
- Blood pressure and cholesterol monitoring
- Kidney function monitoring

Remember: Always consult healthcare professionals for personalized diabetes management plans.
        """,
        
        "hypertension_treatment.txt": """
Hypertension Treatment Protocol

Hypertension (high blood pressure) treatment involves multiple approaches:

1. Lifestyle Modifications
- Sodium reduction (<2,300 mg/day)
- DASH diet (Dietary Approaches to Stop Hypertension)
- Regular aerobic exercise (30 minutes/day)
- Weight management
- Smoking cessation
- Alcohol moderation

2. Medication Classes
- ACE inhibitors: lisinopril, enalapril
- Angiotensin receptor blockers: losartan, valsartan
- Calcium channel blockers: amlodipine, diltiazem
- Diuretics: hydrochlorothiazide, chlorthalidone
- Beta-blockers: metoprolol, atenolol

3. Treatment Goals
- Target BP <130/80 mmHg for most adults
- <140/90 mmHg for older adults
- Regular monitoring and medication adjustment
- Combination therapy when needed

4. Monitoring
- Home blood pressure monitoring
- Regular healthcare provider visits
- 24-hour ambulatory monitoring when indicated
- Target organ damage assessment

5. Emergency Situations
- Hypertensive crisis: BP >180/120 mmHg
- Immediate medical attention required
- IV medications may be necessary
- Hospitalization for severe cases

Important: Treatment should be individualized based on patient characteristics and comorbidities.
        """,
        
        "medication_safety.txt": """
Medication Safety Guidelines

Safe medication use is critical for patient outcomes:

1. Prescription Safety
- Verify patient identity and allergies
- Check for drug interactions
- Confirm appropriate dosing
- Review contraindications
- Monitor for side effects

2. Common Drug Interactions
- Warfarin with NSAIDs (increased bleeding risk)
- Statins with grapefruit juice
- ACE inhibitors with potassium supplements
- Beta-blockers with calcium channel blockers
- Antibiotics with oral contraceptives

3. Side Effect Monitoring
- Regular laboratory monitoring
- Patient education on symptoms
- Reporting systems for adverse events
- Dose adjustments as needed
- Alternative medications when necessary

4. Patient Education
- Proper administration techniques
- Storage requirements
- Timing of doses
- What to do if doses are missed
- Emergency contact information

5. High-Risk Medications
- Anticoagulants (warfarin, heparin)
- Insulin and diabetes medications
- Chemotherapy agents
- Opioids and controlled substances
- Immunosuppressants

Safety First: Always double-check medications and consult healthcare providers for concerns.
        """
    }
    
    # Create data directory
    data_dir = Path("data/uploads")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Write documents
    for filename, content in documents.items():
        file_path = data_dir / filename
        with open(file_path, 'w') as f:
            f.write(content.strip())
        logger.info(f"Created sample document: {filename}")
    
    return list(documents.keys())

def run_document_processing_demo():
    """Demo document processing and vector storage."""
    print("\n" + "="*60)
    print("📄 DOCUMENT PROCESSING DEMO")
    print("="*60)
    
    try:
        from medical_rag.document_processor import MedicalDocumentProcessor
        from medical_rag.vector_store import MedicalVectorStore
        
        # Create sample documents
        document_files = create_sample_medical_documents()
        
        # Initialize components
        processor = MedicalDocumentProcessor()
        vector_store = MedicalVectorStore()
        
        # Process documents
        print("Processing medical documents...")
        all_documents = []
        
        for filename in document_files:
            file_path = f"data/uploads/{filename}"
            documents = processor.process_document(file_path)
            all_documents.extend(documents)
            print(f"  ✓ Processed {filename}: {len(documents)} chunks")
        
        # Add to vector store
        print("Adding documents to vector store...")
        vector_store.add_documents(all_documents)
        
        # Get statistics
        stats = vector_store.get_collection_stats()
        print(f"  ✓ Vector store contains {stats['total_documents']} documents")
        
        return True
        
    except Exception as e:
        logger.error(f"Document processing demo failed: {e}")
        return False

def run_query_demo():
    """Demo medical query processing."""
    print("\n" + "="*60)
    print("🔍 MEDICAL QUERY DEMO")
    print("="*60)
    
    try:
        from medical_rag.vector_store import MedicalVectorStore
        from medical_rag.generation import MedicalResponseGenerator
        
        # Initialize components
        vector_store = MedicalVectorStore()
        generator = MedicalResponseGenerator()
        
        # Sample medical queries
        queries = [
            "What are the key components of diabetes management?",
            "How should hypertension be treated?",
            "What are important medication safety considerations?",
            "What are the side effects of common medications?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query}")
            print("-" * 50)
            
            # Retrieve relevant documents
            retrieved_docs = vector_store.similarity_search(query)
            
            if not retrieved_docs:
                print("  ❌ No relevant documents found")
                continue
            
            print(f"  ✓ Found {len(retrieved_docs)} relevant documents")
            
            # Generate response
            generation_result = generator.generate_response(query, retrieved_docs)
            
            print(f"  📝 Response: {generation_result['response'][:200]}...")
            print(f"  ⏱️  Generation time: {generation_result['generation_time']:.2f}s")
            print(f"  🛡️  Safety score: {generation_result['safety_score']:.3f}")
            
            # Show sources
            if generation_result['sources']:
                print(f"  📚 Sources: {len(generation_result['sources'])} documents")
        
        return True
        
    except Exception as e:
        logger.error(f"Query demo failed: {e}")
        return False

def run_ragas_evaluation_demo():
    """Demo RAGAS evaluation."""
    print("\n" + "="*60)
    print("📊 RAGAS EVALUATION DEMO")
    print("="*60)
    
    try:
        from ragas_framework.evaluation import RAGASEvaluationPipeline
        
        # Initialize evaluation pipeline
        pipeline = RAGASEvaluationPipeline()
        
        # Sample evaluation data
        questions = [
            "What are the key components of diabetes management?",
            "How should hypertension be treated?",
            "What are important medication safety considerations?"
        ]
        
        contexts = [
            ["Diabetes management involves blood sugar monitoring, diet control, exercise, and medication as prescribed by healthcare providers."],
            ["Hypertension treatment involves lifestyle modifications, medication classes, and regular monitoring."],
            ["Medication safety includes prescription verification, drug interaction checking, and side effect monitoring."]
        ]
        
        answers = [
            "Diabetes management includes monitoring blood sugar, following a healthy diet, regular exercise, and taking prescribed medications under medical supervision.",
            "Hypertension treatment involves lifestyle changes, medication therapy, and regular blood pressure monitoring under healthcare provider guidance.",
            "Medication safety requires prescription verification, interaction checking, side effect monitoring, and patient education."
        ]
        
        ground_truths = [
            "Diabetes management requires blood sugar monitoring, dietary control, exercise, and medication adherence.",
            "Hypertension treatment includes lifestyle modifications, medication therapy, and regular monitoring.",
            "Medication safety involves prescription verification, interaction checking, and side effect monitoring."
        ]
        
        print("Running RAGAS evaluation...")
        
        # Run evaluation
        results = pipeline.evaluate_batch(
            questions=questions,
            contexts=contexts,
            answers=answers,
            ground_truths=ground_truths,
            batch_name="demo_evaluation"
        )
        
        # Display results
        print(f"\n📈 Evaluation Results:")
        print(f"  Batch: {results['batch_name']}")
        print(f"  Time: {results['evaluation_time']:.2f}s")
        print(f"  Queries: {results['num_queries']}")
        
        print(f"\n📊 Metrics:")
        for metric, score in results['metrics'].items():
            print(f"  {metric.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"\n✅ Quality Check:")
        quality_check = results['quality_check']
        print(f"  Overall Pass: {quality_check['overall_pass']}")
        
        if quality_check['failed_metrics']:
            print(f"  Failed Metrics: {', '.join(quality_check['failed_metrics'])}")
        
        if quality_check['warnings']:
            print(f"  Warnings: {len(quality_check['warnings'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"RAGAS evaluation demo failed: {e}")
        return False

def run_monitoring_demo():
    """Demo real-time monitoring."""
    print("\n" + "="*60)
    print("📊 MONITORING DEMO")
    print("="*60)
    
    try:
        from ragas_framework.monitoring import RAGASMonitor
        
        # Initialize monitor
        monitor = RAGASMonitor()
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate some evaluation results
        sample_results = [
            {
                "batch_name": "demo_batch_1",
                "timestamp": "2024-01-01T10:00:00",
                "metrics": {
                    "faithfulness": 0.92,
                    "context_precision": 0.88,
                    "context_recall": 0.85,
                    "answer_relevancy": 0.90
                },
                "quality_check": {
                    "overall_pass": True,
                    "failed_metrics": [],
                    "warnings": []
                }
            },
            {
                "batch_name": "demo_batch_2",
                "timestamp": "2024-01-01T10:05:00",
                "metrics": {
                    "faithfulness": 0.78,  # Below threshold
                    "context_precision": 0.82,
                    "context_recall": 0.80,
                    "answer_relevancy": 0.85
                },
                "quality_check": {
                    "overall_pass": False,
                    "failed_metrics": ["faithfulness"],
                    "warnings": ["Warning: faithfulness below threshold (0.78 < 0.90)"]
                }
            }
        ]
        
        # Add results to monitoring
        for result in sample_results:
            monitor.add_evaluation_result(result)
            print(f"  ✓ Added evaluation result: {result['batch_name']}")
        
        # Get current metrics
        current_metrics = monitor.get_current_metrics()
        print(f"\n📊 Current Metrics Status: {current_metrics['status']}")
        
        if current_metrics['status'] == 'active':
            print(f"  Recent Evaluations: {current_metrics['status_info']['recent_evaluations']}")
            print(f"  Total Evaluations: {current_metrics['status_info']['total_evaluations']}")
        
        # Get alerts
        alerts = monitor.get_alerts(hours=1)
        print(f"\n🚨 Alerts: {len(alerts)} in the last hour")
        
        for alert in alerts:
            print(f"  - {alert['severity'].upper()}: {alert.get('warnings', ['Alert triggered'])[0]}")
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        return True
        
    except Exception as e:
        logger.error(f"Monitoring demo failed: {e}")
        return False

def run_standalone_evaluation_demo():
    """Demo standalone RAGAS evaluation script."""
    print("\n" + "="*60)
    print("🔬 STANDALONE EVALUATION DEMO")
    print("="*60)
    
    try:
        # Check if sample dataset exists
        dataset_path = "data/sample_medical_dataset.csv"
        
        if not Path(dataset_path).exists():
            print(f"  ❌ Sample dataset not found: {dataset_path}")
            return False
        
        print("Running standalone RAGAS evaluation...")
        
        # Import and run evaluation
        import subprocess
        import sys
        
        # Run the evaluation script
        result = subprocess.run([
            sys.executable, "ragas_evaluation.py",
            "--dataset", dataset_path,
            "--batch-name", "demo_standalone",
            "--output-dir", "./evaluation_results"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ Standalone evaluation completed successfully")
            print("  📄 Check evaluation_results/ for detailed results")
        else:
            print(f"  ❌ Standalone evaluation failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Standalone evaluation demo failed: {e}")
        return False

def main():
    """Run the complete demo."""
    print("🏥 MEDICAL KNOWLEDGE ASSISTANT RAG SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases the complete medical RAG pipeline with RAGAS evaluation")
    print("=" * 60)
    
    # Check environment
    if not setup_environment():
        print("\n⚠️  Continuing demo with limited functionality...")
    
    # Run demos
    demos = [
        ("Document Processing", run_document_processing_demo),
        ("Medical Queries", run_query_demo),
        ("RAGAS Evaluation", run_ragas_evaluation_demo),
        ("Real-time Monitoring", run_monitoring_demo),
        ("Standalone Evaluation", run_standalone_evaluation_demo)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n🚀 Starting {demo_name} demo...")
            success = demo_func()
            results[demo_name] = success
            
            if success:
                print(f"  ✅ {demo_name} demo completed successfully")
            else:
                print(f"  ❌ {demo_name} demo failed")
                
        except Exception as e:
            logger.error(f"{demo_name} demo failed: {e}")
            results[demo_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📋 DEMO SUMMARY")
    print("="*60)
    
    successful_demos = sum(results.values())
    total_demos = len(results)
    
    for demo_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {demo_name}: {status}")
    
    print(f"\nOverall: {successful_demos}/{total_demos} demos completed successfully")
    
    if successful_demos == total_demos:
        print("\n🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("  1. Set your OpenAI API key: export OPENAI_API_KEY='your_key_here'")
        print("  2. Start the API server: python app.py")
        print("  3. Access the dashboard: http://localhost:8000/dashboard")
        print("  4. Try the API endpoints: http://localhost:8000/docs")
    else:
        print(f"\n⚠️  {total_demos - successful_demos} demo(s) failed. Check the logs above for details.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main() 