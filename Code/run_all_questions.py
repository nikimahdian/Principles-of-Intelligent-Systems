"""
Main Runner Script for FIS4041 Final Project
Executes all questions and stores results for LaTeX report generation
"""

import sys
import os
import json
from datetime import datetime

# Add Code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_question(question_num, module_name, description):
    """Run a question module and handle errors"""
    print("\n" + "=" * 80)
    print(f"RUNNING {description}")
    print("=" * 80)
    
    try:
        # Import and run the question module
        module = __import__(module_name)
        if hasattr(module, 'main'):
            module.main()
            print(f"\n✓ {description} completed successfully!")
            return True
        else:
            print(f"\n✗ {description} module does not have a main() function")
            return False
    except Exception as e:
        print(f"\n✗ Error running {description}:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_summary_report():
    """Create a summary report of all executed questions"""
    summary = {
        'execution_time': datetime.now().isoformat(),
        'questions': {},
        'files_generated': {
            'results': [],
            'explanations': [],
            'plots': []
        }
    }
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Check for generated files
    results_dir = os.path.join(project_root, 'Results')
    explanations_dir = os.path.join(project_root, 'Explanations')
    
    if os.path.exists(results_dir):
        summary['files_generated']['results'] = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if os.path.exists(explanations_dir):
        summary['files_generated']['explanations'] = [f for f in os.listdir(explanations_dir) if f.endswith('.json')]
    
    if os.path.exists(results_dir):
        summary['files_generated']['plots'] = [f for f in os.listdir(results_dir) if f.endswith('.png')]
    
    # Save summary
    summary_path = os.path.join(results_dir, 'execution_summary.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4, default=str)
    
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Execution Time: {summary['execution_time']}")
    print(f"\nResults Files: {len(summary['files_generated']['results'])}")
    for f in summary['files_generated']['results']:
        print(f"  - {f}")
    print(f"\nExplanation Files: {len(summary['files_generated']['explanations'])}")
    for f in summary['files_generated']['explanations']:
        print(f"  - {f}")
    print(f"\nPlot Files: {len(summary['files_generated']['plots'])}")
    for f in summary['files_generated']['plots']:
        print(f"  - {f}")
    print("\n" + "=" * 80)

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("FIS4041 FINAL PROJECT - COMPREHENSIVE EXECUTION")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    questions = [
        (1, 'question1_feature_selection', 'Question 1: Feature Selection (PSO & GA)'),
        (2, 'question2_clustering', 'Question 2: Clustering (KMeans, Agglomerative, DBSCAN)'),
        (3, 'question3_qlearning', 'Question 3: Q-Learning Theory'),
        ('FP', 'final_project_ml_pipeline', 'Final Project: Comprehensive ML Pipeline'),
    ]
    
    results = {}
    
    for q_num, module_name, description in questions:
        success = run_question(q_num, module_name, description)
        results[f'question_{q_num}'] = {
            'module': module_name,
            'description': description,
            'success': success
        }
    
    # Create summary report
    create_summary_report()
    
    print("\n" + "=" * 80)
    print("ALL QUESTIONS EXECUTION COMPLETED")
    print("=" * 80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext Steps:")
    print("1. Review results in the Results/ directory")
    print("2. Review explanations in the Explanations/ directory")
    print("3. Check generated plots in the Results/ directory")
    print("4. Use the explanations JSON files to generate LaTeX report")
    print("=" * 80)

if __name__ == "__main__":
    main()

