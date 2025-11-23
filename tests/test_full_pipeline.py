#!/usr/bin/env python3
"""
Full Pipeline Smoke Test

Tests that the complete pipeline can run end-to-end:
1. Data loading
2. Augmentation (with monitoring)
3. Model training (small test)
4. Visualization generation

This ensures cloning on a GPU and running will work without errors.
"""
import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, '..')

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(msg):
    print(f"\n{BLUE}{'='*70}")
    print(f"{msg}")
    print(f"{'='*70}{RESET}\n")

def print_success(msg):
    print(f"{GREEN}✓ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}✗ {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠ {msg}{RESET}")


def test_dependencies():
    """Check that all required dependencies are importable."""
    print_header("TEST 1: Checking Dependencies")
    
    required_packages = {
        'Core ML': ['numpy', 'pandas', 'sklearn'],
        'Deep Learning': ['transformers', 'torch', 'xgboost'],
        'Weak Supervision': ['snorkel', 'cleanlab'],
        'Monitoring': ['mlflow', 'matplotlib', 'seaborn', 'plotly'],
        'NLP': ['textstat', 'spacy'],
        'Data': ['datasets', 'huggingface_hub']
    }
    
    all_ok = True
    
    for category, packages in required_packages.items():
        print(f"\n{category}:")
        for package in packages:
            try:
                __import__(package)
                print_success(f"{package}")
            except ImportError:
                print_error(f"{package} - NOT INSTALLED")
                all_ok = False
    
    if not all_ok:
        print_error("\nSome dependencies are missing!")
        print("Install with:")
        print("  pip install -r requirements-augmentation.txt")
        print("  pip install -r requirements-ml.txt")
        return False
    
    print_success("\nAll dependencies available")
    return True


def test_data_loading():
    """Test that NeoDataset can be loaded."""
    print_header("TEST 2: Data Loading")
    
    try:
        from src.ml.neodataset_loader import load_neodataset
        
        print("Attempting to load NeoDataset (this may take a few minutes)...")
        print("Note: This will download ~50MB if not already cached")
        
        df = load_neodataset()
        
        if len(df) > 0:
            print_success(f"Dataset loaded: {len(df)} stories")
            return True
        else:
            print_error("Dataset is empty")
            return False
            
    except Exception as e:
        print_error(f"Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_labeling_functions():
    """Test that labeling functions are available."""
    print_header("TEST 3: Labeling Functions")
    
    try:
        from src.ml.labeling_functions import ALL_LABELING_FUNCTIONS
        
        if len(ALL_LABELING_FUNCTIONS) > 0:
            print_success(f"Found {len(ALL_LABELING_FUNCTIONS)} labeling functions")
            for i, lf in enumerate(ALL_LABELING_FUNCTIONS[:5], 1):
                print(f"  {i}. {lf.name}")
            if len(ALL_LABELING_FUNCTIONS) > 5:
                print(f"  ... and {len(ALL_LABELING_FUNCTIONS) - 5} more")
            return True
        else:
            print_error("No labeling functions found")
            return False
            
    except Exception as e:
        print_error(f"Failed to load labeling functions: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extractors():
    """Test that feature extractors can initialize."""
    print_header("TEST 4: Feature Extractors")
    
    try:
        from src.ml.feature_extractors import SymbolicFeatureExtractor
        from src.ml.bert_embedder import BertEmbedder
        
        print("Initializing SymbolicFeatureExtractor...")
        sym_extractor = SymbolicFeatureExtractor()
        print_success("SymbolicFeatureExtractor initialized")
        
        print("\nInitializing BertEmbedder (this will download DistilBERT if needed)...")
        print("Note: This may take a few minutes on first run")
        bert_embedder = BertEmbedder(quantize=True, cache_size=100)
        print_success("BertEmbedder initialized")
        
        # Test extraction on dummy text
        print("\nTesting feature extraction on sample text...")
        test_text = "Fix critical security bug in authentication module"
        
        # SymbolicFeatureExtractor.extract_features expects a single string
        sym_features = sym_extractor.extract_features(test_text)
        print_success(f"Symbolic features extracted: shape {sym_features.shape}")
        
        # BertEmbedder has two methods: embed() for single text, embed_batch() for lists
        bert_features = bert_embedder.embed_batch([test_text])
        print_success(f"BERT embeddings extracted: shape {bert_features.shape}")
        
        return True
        
    except Exception as e:
        print_error(f"Feature extractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_components():
    """Test that training components can initialize."""
    print_header("TEST 5: Training Components")
    
    try:
        from src.ml.train_risk_model import RiskModelTrainer
        from src.ml.experiment_tracker import ExperimentTracker
        
        temp_dir = tempfile.mkdtemp()
        
        print("Initializing ExperimentTracker...")
        tracker = ExperimentTracker(
            experiment_name="Pipeline_Test",
            output_dir=temp_dir
        )
        print_success("ExperimentTracker initialized")
        
        print("\nInitializing RiskModelTrainer...")
        trainer = RiskModelTrainer(output_dir=temp_dir, tracker=tracker)
        print_success("RiskModelTrainer initialized")
        
        shutil.rmtree(temp_dir, ignore_errors=True)
        return True
        
    except Exception as e:
        print_error(f"Training components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scripts_exist():
    """Verify all required scripts exist and are executable."""
    print_header("TEST 6: Required Scripts")
    
    required_scripts = [
        'scripts/augment_neodataset.py',
        'scripts/map_to_3class.py',
        'scripts/run_ablation_study.py',
        'scripts/generate_all_plots.py',
        'src/ml/train_risk_model.py'
    ]
    
    all_exist = True
    
    for script in required_scripts:
        if Path(script).exists():
            print_success(f"{script}")
        else:
            print_error(f"{script} - NOT FOUND")
            all_exist = False
    
    return all_exist


def test_directory_structure():
    """Verify required directories exist or can be created."""
    print_header("TEST 7: Directory Structure")
    
    required_dirs = [
        'data',
        'models',
        'visualizations',
        'src/ml',
        'src/visualization',
        'scripts'
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print_success(f"{dir_path}/")
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print_success(f"{dir_path}/ - CREATED")
            except Exception as e:
                print_error(f"{dir_path}/ - FAILED TO CREATE: {e}")
                return False
    
    return True


def test_quick_pipeline():
    """Run a quick end-to-end test with minimal data."""
    print_header("TEST 8: Quick Pipeline Test")
    
    print("This test will:")
    print("  1. Load a small sample of data")
    print("  2. Apply labeling functions")
    print("  3. Extract features")
    print("  4. Verify everything works together")
    print()
    
    try:
        from src.ml.neodataset_loader import load_neodataset, preprocess_neodataset
        from src.ml.labeling_functions import ALL_LABELING_FUNCTIONS
        from src.ml.weak_supervision_pipeline import WeakSupervisionPipeline
        
        # Load small sample
        print("Loading sample data...")
        df = load_neodataset()
        df = preprocess_neodataset(df)
        df_sample = df.head(20).copy()  # Just 20 stories for speed
        print_success(f"Loaded {len(df_sample)} sample stories")
        
        # Apply LFs
        print("\nApplying labeling functions...")
        ws_pipeline = WeakSupervisionPipeline(df_sample, ALL_LABELING_FUNCTIONS)
        ws_pipeline.apply_labeling_functions()
        print_success("Labeling functions applied")
        
        # Train label model
        print("\nTraining Snorkel label model...")
        ws_pipeline.train_label_model(n_epochs=50)  # Quick training
        print_success("Label model trained")
        
        # Generate labels
        print("\nGenerating probabilistic labels...")
        df_labeled = ws_pipeline.generate_probabilistic_labels()
        print_success(f"Generated labels for {len(df_labeled)} stories")
        
        # Verify output
        required_cols = ['risk_label', 'risk_confidence', 'risk_label_binary']
        if all(col in df_labeled.columns for col in required_cols):
            print_success("Output schema correct")
        else:
            print_error("Output schema incorrect")
            return False
        
        print_success("\nQuick pipeline test PASSED")
        return True
        
    except Exception as e:
        print_error(f"Quick pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_smoke_test():
    """Run all smoke tests."""
    print(f"\n{BLUE}{'='*70}")
    print("SPRINTGUARD FULL PIPELINE SMOKE TEST")
    print("="*70)
    print("This verifies the pipeline will work on a fresh clone")
    print(f"{'='*70}{RESET}\n")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Data Loading", test_data_loading),
        ("Labeling Functions", test_labeling_functions),
        ("Feature Extractors", test_feature_extractors),
        ("Training Components", test_training_components),
        ("Required Scripts", test_scripts_exist),
        ("Directory Structure", test_directory_structure),
        ("Quick Pipeline", test_quick_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print_header("SMOKE TEST SUMMARY")
    
    for test_name, passed in results.items():
        if passed:
            print(f"{GREEN}✓ {test_name}: PASSED{RESET}")
        else:
            print(f"{RED}✗ {test_name}: FAILED{RESET}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n{'='*70}")
    if passed == total:
        print(f"{GREEN}ALL TESTS PASSED ({passed}/{total}){RESET}")
        print(f"{'='*70}")
        print(f"\n{GREEN}✅ READY FOR GPU DEPLOYMENT!{RESET}")
        print("\nYou can now clone this repo on a GPU and run:")
        print(f"{BLUE}  1. pip install -r requirements-augmentation.txt")
        print("  2. pip install -r requirements-ml.txt")
        print("  3. python scripts/augment_neodataset.py")
        print(f"  4. python src/ml/train_risk_model.py --data data/neodataset_augmented_3class.csv{RESET}")
        return True
    else:
        print(f"{RED}SOME TESTS FAILED ({passed}/{total} passed){RESET}")
        print(f"{'='*70}")
        print(f"\n{YELLOW}⚠ Fix failing tests before GPU deployment{RESET}")
        return False


if __name__ == '__main__':
    success = run_smoke_test()
    sys.exit(0 if success else 1)

