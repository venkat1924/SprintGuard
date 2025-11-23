#!/usr/bin/env python3
"""
Test Suite for ML Monitoring Infrastructure

Tests all new monitoring components to ensure they work correctly.
"""
import sys
import os
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, '..')

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_test(name):
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print('='*70)

def print_success(msg):
    print(f"{GREEN}✓ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}✗ {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠ {msg}{RESET}")


def test_imports():
    """Test that all new modules can be imported."""
    print_test("Import Tests")
    
    try:
        from src.ml.experiment_tracker import ExperimentTracker
        print_success("ExperimentTracker imported")
    except Exception as e:
        print_error(f"Failed to import ExperimentTracker: {e}")
        return False
    
    try:
        from src.visualization.publication_plots import (
            generate_sankey_diagram,
            generate_ablation_study,
            generate_lf_correlation_heatmap,
            generate_calibration_plot,
            generate_tsne_embeddings
        )
        print_success("Publication plots module imported")
    except Exception as e:
        print_error(f"Failed to import publication plots: {e}")
        return False
    
    # Test optional dependencies
    try:
        import mlflow
        print_success(f"MLflow version: {mlflow.__version__}")
    except ImportError:
        print_warning("MLflow not installed (required)")
        return False
    
    try:
        import plotly
        print_success(f"Plotly version: {plotly.__version__}")
    except ImportError:
        print_warning("Plotly not installed (optional for Sankey)")
    
    try:
        import matplotlib.pyplot as plt
        plt.style.use(['science', 'ieee'])
        print_success("SciencePlots available")
    except:
        print_warning("SciencePlots not installed (optional, will use defaults)")
    
    return True


def test_experiment_tracker():
    """Test ExperimentTracker functionality."""
    print_test("ExperimentTracker Tests")
    
    from src.ml.experiment_tracker import ExperimentTracker
    
    # Create temporary output directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test initialization
        tracker = ExperimentTracker(
            experiment_name="Test_Experiment",
            output_dir=temp_dir
        )
        print_success("ExperimentTracker initialized")
        
        # Test start_run
        tracker.start_run(run_name="test_run", tags={"test": "true"})
        print_success("Run started")
        
        # Test log_params
        tracker.log_params({"param1": "value1", "param2": 42})
        print_success("Parameters logged")
        
        # Test log_metric
        tracker.log_metric("test_metric", 0.95)
        print_success("Single metric logged")
        
        # Test log_stage_metrics
        tracker.log_stage_metrics("test_stage", {
            "metric1": 0.8,
            "metric2": 0.9,
            "metric3": 100
        })
        print_success("Stage metrics logged")
        
        # Test log_stage_count
        tracker.log_stage_count("raw", 12000)
        tracker.log_stage_count("filtered", 10000)
        print_success("Stage counts logged")
        
        # Test log_figure
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title("Test Plot")
        tracker.log_figure(fig, "test_plot")
        print_success("Figure logged")
        
        # Verify files were created
        pdf_path = Path(temp_dir) / "test_plot.pdf"
        svg_path = Path(temp_dir) / "test_plot.svg"
        if pdf_path.exists() and svg_path.exists():
            print_success("PDF and SVG files created")
        else:
            print_error("Plot files not created")
        
        # Test end_run
        tracker.end_run()
        print_success("Run ended")
        
        return True
        
    except Exception as e:
        print_error(f"ExperimentTracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_publication_plots():
    """Test publication plot generators."""
    print_test("Publication Plots Tests")
    
    from src.visualization.publication_plots import (
        generate_ablation_study,
        generate_lf_correlation_heatmap,
        generate_calibration_plot,
        generate_tsne_embeddings
    )
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test 1: Ablation Study
        results = {
            'Baseline': (0.72, 0.03),
            'Method A': (0.78, 0.02),
            'Method B': (0.83, 0.02)
        }
        output_path = os.path.join(temp_dir, "ablation_test.pdf")
        fig = generate_ablation_study(results, output_path=output_path)
        if os.path.exists(output_path):
            print_success("Ablation study plot generated")
        else:
            print_error("Ablation study plot not created")
        
        # Test 2: LF Correlation Heatmap
        L_matrix = np.random.randint(-1, 2, size=(100, 10))  # 100 samples, 10 LFs
        lf_names = [f"LF_{i}" for i in range(10)]
        output_path = os.path.join(temp_dir, "heatmap_test.pdf")
        fig = generate_lf_correlation_heatmap(L_matrix, lf_names, output_path=output_path)
        if os.path.exists(output_path):
            print_success("LF correlation heatmap generated")
        else:
            print_error("LF heatmap not created")
        
        # Test 3: Calibration Plot
        n_samples = 200
        y_true = np.random.randint(0, 3, size=n_samples)
        y_pred_proba = np.random.dirichlet(np.ones(3), size=n_samples)
        output_path = os.path.join(temp_dir, "calibration_test.pdf")
        fig = generate_calibration_plot(y_true, y_pred_proba, output_path=output_path)
        if os.path.exists(output_path):
            print_success("Calibration plot generated")
        else:
            print_error("Calibration plot not created")
        
        # Test 4: t-SNE Embeddings
        embeddings = np.random.randn(300, 50)  # 300 samples, 50-dim embeddings
        labels = np.random.randint(0, 3, size=300)
        output_path = os.path.join(temp_dir, "tsne_test.pdf")
        fig = generate_tsne_embeddings(embeddings, labels, output_path=output_path, sample_size=300)
        if os.path.exists(output_path):
            print_success("t-SNE plot generated")
        else:
            print_error("t-SNE plot not created")
        
        return True
        
    except Exception as e:
        print_error(f"Publication plots test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_sankey_diagram():
    """Test Sankey diagram generation (requires plotly)."""
    print_test("Sankey Diagram Test")
    
    try:
        import plotly
        from src.visualization.publication_plots import generate_sankey_diagram
        
        temp_dir = tempfile.mkdtemp()
        
        stage_counts = {
            'raw': 12106,
            'snorkel': 12106,
            'cleanlab': 10500,
            'final': 9000
        }
        
        output_path = os.path.join(temp_dir, "sankey_test.pdf")
        fig = generate_sankey_diagram(stage_counts, output_path=output_path)
        
        # Check if HTML was created (PDF requires kaleido)
        html_path = output_path.replace('.pdf', '.html')
        if os.path.exists(html_path):
            print_success("Sankey diagram HTML generated")
        else:
            print_warning("Sankey HTML not created")
        
        # PDF may not be created if kaleido is missing
        if os.path.exists(output_path):
            print_success("Sankey diagram PDF generated")
        else:
            print_warning("Sankey PDF not created (kaleido may be missing)")
        
        shutil.rmtree(temp_dir, ignore_errors=True)
        return True
        
    except ImportError:
        print_warning("Plotly not installed, skipping Sankey test")
        return True  # Not a failure, just not available
    except Exception as e:
        print_error(f"Sankey diagram test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_integration():
    """Test that modified pipeline components work."""
    print_test("Pipeline Integration Tests")
    
    try:
        # Test WeakSupervisionPipeline additions
        from src.ml.weak_supervision_pipeline import WeakSupervisionPipeline
        print_success("WeakSupervisionPipeline imported")
        
        # Check if log_lf_diagnostics method exists
        if hasattr(WeakSupervisionPipeline, 'log_lf_diagnostics'):
            print_success("WeakSupervisionPipeline.log_lf_diagnostics exists")
        else:
            print_error("WeakSupervisionPipeline.log_lf_diagnostics not found")
            return False
        
        # Test CleanlabPipeline additions
        from src.ml.cleanlab_pipeline import CleanlabPipeline
        print_success("CleanlabPipeline imported")
        
        if hasattr(CleanlabPipeline, 'log_cleanlab_diagnostics'):
            print_success("CleanlabPipeline.log_cleanlab_diagnostics exists")
        else:
            print_error("CleanlabPipeline.log_cleanlab_diagnostics not found")
            return False
        
        # Test RiskModelTrainer additions
        from src.ml.train_risk_model import RiskModelTrainer
        print_success("RiskModelTrainer imported")
        
        if hasattr(RiskModelTrainer, 'visualize_embeddings'):
            print_success("RiskModelTrainer.visualize_embeddings exists")
        else:
            print_error("RiskModelTrainer.visualize_embeddings not found")
            return False
        
        # Test that tracker parameter exists in __init__
        import inspect
        sig = inspect.signature(RiskModelTrainer.__init__)
        if 'tracker' in sig.parameters:
            print_success("RiskModelTrainer accepts tracker parameter")
        else:
            print_error("RiskModelTrainer missing tracker parameter")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scripts_executable():
    """Test that new scripts are executable and parseable."""
    print_test("Script Validation Tests")
    
    scripts = [
        'scripts/run_ablation_study.py',
        'scripts/generate_all_plots.py'
    ]
    
    all_ok = True
    
    for script_path in scripts:
        try:
            with open(script_path, 'r') as f:
                code = f.read()
            
            # Try to compile the script
            compile(code, script_path, 'exec')
            print_success(f"{script_path} is valid Python")
            
            # Check if executable
            if os.access(script_path, os.X_OK):
                print_success(f"{script_path} is executable")
            else:
                print_warning(f"{script_path} not executable (not critical)")
                
        except SyntaxError as e:
            print_error(f"{script_path} has syntax error: {e}")
            all_ok = False
        except FileNotFoundError:
            print_error(f"{script_path} not found")
            all_ok = False
        except Exception as e:
            print_error(f"{script_path} validation failed: {e}")
            all_ok = False
    
    return all_ok


def test_weak_supervision_mock():
    """Test WeakSupervisionPipeline.log_lf_diagnostics with mock data."""
    print_test("WeakSupervisionPipeline Logging Test")
    
    try:
        from src.ml.weak_supervision_pipeline import WeakSupervisionPipeline
        from src.ml.experiment_tracker import ExperimentTracker
        
        # Create mock data
        df = pd.DataFrame({
            'full_text': ['test story ' + str(i) for i in range(50)],
            'word_count': np.random.randint(10, 100, 50),
            'story_points': np.random.randint(1, 10, 50)
        })
        
        # Create mock labeling functions
        from snorkel.labeling import labeling_function
        
        @labeling_function()
        def lf_test_1(x):
            return 1 if 'test' in x.full_text else -1
        
        @labeling_function()
        def lf_test_2(x):
            return 0 if len(x.full_text) < 20 else -1
        
        lfs = [lf_test_1, lf_test_2]
        
        # Create pipeline
        ws_pipeline = WeakSupervisionPipeline(df, lfs)
        ws_pipeline.apply_labeling_functions()
        
        # Create tracker
        temp_dir = tempfile.mkdtemp()
        tracker = ExperimentTracker(
            experiment_name="Test_WS",
            output_dir=temp_dir
        )
        tracker.start_run(run_name="test_ws_logging")
        
        # Test logging
        ws_pipeline.log_lf_diagnostics(tracker)
        
        tracker.end_run()
        
        print_success("WeakSupervisionPipeline logging works")
        
        shutil.rmtree(temp_dir, ignore_errors=True)
        return True
        
    except Exception as e:
        print_error(f"WeakSupervisionPipeline logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("SPRINTGUARD MONITORING TEST SUITE")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("ExperimentTracker", test_experiment_tracker),
        ("Publication Plots", test_publication_plots),
        ("Sankey Diagram", test_sankey_diagram),
        ("Pipeline Integration", test_pipeline_integration),
        ("Script Validation", test_scripts_executable),
        ("WS Pipeline Logging", test_weak_supervision_mock)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        if passed:
            print(f"{GREEN}✓ {test_name}: PASSED{RESET}")
        else:
            print(f"{RED}✗ {test_name}: FAILED{RESET}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "="*70)
    if passed == total:
        print(f"{GREEN}ALL TESTS PASSED ({passed}/{total}){RESET}")
        print("="*70)
        print("\n✅ Monitoring infrastructure is ready to use!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements-ml.txt")
        print("  2. Run pipeline: python scripts/augment_neodataset.py")
        print("  3. View results: mlflow ui")
        return True
    else:
        print(f"{RED}SOME TESTS FAILED ({passed}/{total} passed){RESET}")
        print("="*70)
        print("\n⚠ Please fix the failing tests before proceeding.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

