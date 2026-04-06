"""
Utility functions for MT benchmarking
"""
import os
import sys
import json
import yaml
import psutil
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import torch
import cpuinfo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_hardware_info() -> Dict:
    """Detect and return hardware capabilities"""
    info = {
        'cpu': {
            'cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'model': cpuinfo.get_cpu_info().get('brand_raw', 'Unknown'),
            'freq_ghz': psutil.cpu_freq().max / 1000 if psutil.cpu_freq() else None
        },
        'ram': {
            'total_gb': psutil.virtual_memory().total / (1024**3),
            'available_gb': psutil.virtual_memory().available / (1024**3)
        },
        'gpu': {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'devices': []
        },
        'os': sys.platform
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info = {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'memory_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3)
            }
            info['gpu']['devices'].append(gpu_info)
    
    return info


def recommend_dtype_and_batch_size(hardware_info: Dict) -> Tuple[str, int]:
    """Recommend dtype and batch size based on hardware"""
    if not hardware_info['gpu']['available']:
        logger.warning("No GPU detected. Will use CPU (very slow ~3-4 hours expected)")
        return "float32", 1
    
    gpu_memory_gb = hardware_info['gpu']['devices'][0]['memory_gb']
    
    if gpu_memory_gb < 4:
        logger.warning(f"Low GPU VRAM ({gpu_memory_gb}GB). Using int8 quantization + batch_size=1")
        return "int8", 1
    elif gpu_memory_gb < 8:
        logger.warning(f"Moderate GPU VRAM ({gpu_memory_gb}GB). Using float16 + batch_size=4")
        return "float16", 4
    else:
        logger.info(f"Good GPU VRAM ({gpu_memory_gb}GB). Using float16 + batch_size=8")
        return "float16", 8


def print_hardware_info(hardware_info: Dict):
    """Pretty print hardware information"""
    print("\n" + "="*60)
    print("HARDWARE CONFIGURATION")
    print("="*60)
    
    # CPU
    print(f"\nCPU: {hardware_info['cpu']['model']}")
    print(f"  Cores: {hardware_info['cpu']['cores']} physical, {hardware_info['cpu']['logical_cores']} logical")
    if hardware_info['cpu']['freq_ghz']:
        print(f"  Frequency: {hardware_info['cpu']['freq_ghz']:.2f} GHz")
    
    # RAM
    print(f"\nMemory:")
    print(f"  Total: {hardware_info['ram']['total_gb']:.2f} GB")
    print(f"  Available: {hardware_info['ram']['available_gb']:.2f} GB")
    
    # GPU
    print(f"\nGPU:")
    if hardware_info['gpu']['available']:
        print(f"  Detected: {hardware_info['gpu']['device_count']} device(s)")
        for gpu in hardware_info['gpu']['devices']:
            print(f"  Device {gpu['index']}: {gpu['name']} ({gpu['memory_gb']:.2f} GB)")
    else:
        print("  Not available (CPU-only mode)")
    
    print("\n" + "="*60 + "\n")


def check_model_memory_feasibility(model_config: Dict, hardware_info: Dict, dtype: str = "float16") -> Tuple[bool, str]:
    """Check if model fits in available memory"""
    key = f"vram_{dtype.replace('.', '').lower()}_gb" if dtype != "float32" else "vram_fp32_gb"
    if key not in model_config and dtype == "float32":
        key = "vram_fp32_gb"
    
    required_vram = model_config.get(key, model_config.get("vram_fp16_gb", 0))
    
    if not hardware_info['gpu']['available']:
        # CPU path always possible but slow
        return True, f"Using CPU (model requires {required_vram:.1f}GB VRAM if using GPU)"
    
    available_vram = hardware_info['gpu']['devices'][0]['memory_gb'] * 0.85  # Use 85% of available
    
    if required_vram > available_vram:
        return False, f"Model requires {required_vram:.1f}GB but only {available_vram:.2f}GB available"
    
    return True, f"Feasible (requires {required_vram:.1f}GB, available {available_vram:.2f}GB)"


def setup_directories():
    """Ensure all required directories exist"""
    dirs = [
        './data/cache',
        './data/raw',
        './results',
        './logs'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def clear_gpu_cache():
    """Clear GPU cache between model loads"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class MetricsCache:
    """Simple in-memory cache for loaded metrics"""
    def __init__(self):
        self._cache = {}
    
    def get_or_load(self, metric_name: str):
        """Lazy load metric if not cached"""
        if metric_name not in self._cache:
            logger.info(f"Loading metric: {metric_name}")
            try:
                import evaluate as hf_evaluate
                self._cache[metric_name] = hf_evaluate.load(metric_name)
            except Exception as e:
                logger.error(f"Failed to load metric {metric_name}: {e}")
                self._cache[metric_name] = None
        return self._cache[metric_name]
