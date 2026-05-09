"""
Quick test script to validate benchmarking pipeline
"""
import logging
import sys
from scripts.utils import load_config, get_hardware_info, print_hardware_info
from scripts.data_loader import MTDataLoader
from scripts.evaluator import MTEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_hardware():
    """Test hardware detection"""
    logger.info("Testing hardware detection...")
    hw = get_hardware_info()
    print_hardware_info(hw)
    return hw


def test_config_loading():
    """Test configuration loading"""
    logger.info("Testing config loading...")
    models_cfg = load_config("config/models.yaml")
    datasets_cfg = load_config("config/datasets.yaml")

    logger.info(f"Loaded {len(models_cfg['models'])} models")
    logger.info(f"Models: {list(models_cfg['models'].keys())}")
    return models_cfg, datasets_cfg


def test_data_loading():
    """Test GAC dataset loading"""
    logger.info("Testing data loading (GAC test set)...")

    try:
        loader = MTDataLoader()
        eng, kik = loader.load_gac_test_set("data/selectpairs500.xlsx")
        logger.info(f"Loaded {len(eng)} parallel sentences from GAC test set")

        # Print sample
        logger.info("Sample parallel sentences:")
        for i in range(min(3, len(eng))):
            logger.info(f"  [EN] {eng[i][:60]}...")
            logger.info(f"  [KIK] {kik[i][:60]}...")

        return eng, kik
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        return None, None


def test_metrics():
    """Test metrics computation (CPU-only, no model loading)"""
    logger.info("Testing metrics (BLEU + chrF++)...")

    try:
        evaluator = MTEvaluator()

        # Test with dummy data
        predictions = ["Hello world", "This is a test"]
        references = ["Hello world", "This is a test"]

        # Only test the CPU-bound metrics (no BERTScore/AfriCOMET/Goldfish)
        bleu = evaluator._compute_bleu(predictions, references)
        chrf = evaluator._compute_chrf_pp(predictions, references)
        logger.info(f"BLEU: {bleu}, chrF++: {chrf}")
        return True
    except Exception as e:
        logger.error(f"Metrics test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("MT BENCHMARKING - VALIDATION TESTS")
    logger.info("=" * 60 + "\n")

    tests_passed = 0
    tests_total = 0

    # Test 1: Hardware
    tests_total += 1
    try:
        hw = test_hardware()
        tests_passed += 1
        logger.info("✓ Hardware test passed\n")
    except Exception as e:
        logger.error(f"✗ Hardware test failed: {e}\n")

    # Test 2: Config loading
    tests_total += 1
    try:
        models_cfg, datasets_cfg = test_config_loading()
        tests_passed += 1
        logger.info("✓ Config loading test passed\n")
    except Exception as e:
        logger.error(f"✗ Config test failed: {e}\n")

    # Test 3: Data loading
    tests_total += 1
    try:
        eng, kik = test_data_loading()
        if eng and kik:
            tests_passed += 1
            logger.info("✓ Data loading test passed\n")
        else:
            logger.error("✗ Data loading test returned empty\n")
    except Exception as e:
        logger.error(f"✗ Data loading test failed: {e}\n")

    # Test 4: Metrics
    tests_total += 1
    try:
        if test_metrics():
            tests_passed += 1
            logger.info("✓ Metrics test passed\n")
        else:
            logger.error("✗ Metrics test returned false\n")
    except Exception as e:
        logger.error(f"✗ Metrics test failed: {e}\n")

    # Summary
    logger.info("=" * 60)
    logger.info(f"RESULTS: {tests_passed}/{tests_total} tests passed")
    logger.info("=" * 60)

    if tests_passed == tests_total:
        logger.info("\n✓ All validation tests passed! Ready to run benchmarking.\n")
        logger.info("Run: python main.py --help")
        return 0
    else:
        logger.error(
            f"\n✗ {tests_total - tests_passed} test(s) failed. Fix issues before benchmarking.\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
