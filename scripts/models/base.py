"""
Abstract base class for Machine Translation models
Defines interface that all model implementations must follow
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import logging
import time
import torch

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base for all MT models"""
    
    def __init__(self, config: Dict, device_map: str = "auto"):
        """
        Initialize model with config
        
        Args:
            config: Model configuration dict from models.yaml
            device_map: Device placement strategy ('auto', 'cpu', 'cuda:0', etc.)
        """
        self.config = config
        self.device_map = device_map
        self.model = None
        self.tokenizer = None
        self.model_id = config.get('model_id')
        self.name = config.get('name', self.model_id)
        self.dtype = config.get('dtype', 'torch.float16')
        self._load_time = None
        self._is_loaded = False
    
    @abstractmethod
    def load(self):
        """Load model and tokenizer from disk/HF hub"""
        pass
    
    @abstractmethod
    def translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Translate list of texts
        
        Args:
            texts: List of source language texts
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Override default batch size
            **kwargs: Model-specific parameters
        
        Returns:
            List of translated texts (same length as input)
        """
        pass
    
    def unload(self):
        """Unload model from memory"""
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Unloaded {self.name}")
    
    def get_load_time(self) -> float:
        """Return model load time in seconds"""
        return self._load_time or 0.0
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded"""
        return self._is_loaded
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, loaded={self.is_loaded})"


class TranslationResult:
    """Wrapper for translation results with metadata"""
    
    def __init__(self, translations: List[str], model_name: str, 
                 source_lang: str, target_lang: str, inference_time: float = 0.0):
        self.translations = translations
        self.model_name = model_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.inference_time = inference_time
        self.num_tokens = sum(len(t.split()) for t in translations)
        self.avg_time_per_sentence = inference_time / len(translations) if translations else 0.0
    
    @property
    def speed(self) -> float:
        """Tokens per second"""
        if self.inference_time > 0:
            return self.num_tokens / self.inference_time
        return 0.0
    
    def __repr__(self):
        return f"TranslationResult({self.model_name}, {len(self.translations)} sents, {self.speed:.1f} tok/s)"
