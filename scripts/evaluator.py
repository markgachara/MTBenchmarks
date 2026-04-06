"""
Metric computation for MT evaluation
"""
import logging
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class MTEvaluator:
    """Compute evaluation metrics for MT"""
    
    def __init__(self):
        self.metrics = {}
        self._load_metrics()
    
    def _load_metrics(self):
        """Load evaluation metrics"""
        try:
            import evaluate as hf_evaluate
            
            logger.info("Loading metrics...")
            self.metrics['bleu'] = hf_evaluate.load('bleu')
            self.metrics['chrf'] = hf_evaluate.load('chrf')
            
            # Optional: BERTScore (requires more resources)
            try:
                self.metrics['bertscore'] = hf_evaluate.load('bertscore')
                logger.info("BERTScore loaded (GPU may be used)")
            except Exception as e:
                logger.warning(f"BERTScore not available: {e}")
                self.metrics['bertscore'] = None
            
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
        language_pair: str = "kik-eng"
    ) -> Dict[str, float]:
        """
        Compute all available metrics
        
        Args:
            predictions: List of machine translations
            references: List of reference translations
            language_pair: Language pair code (for sacrebleu tokenization)
        
        Returns:
            Dict with metric_name -> score
        """
        results = {}
        
        # Handle empty predictions
        if not predictions or not references:
            logger.warning("Empty predictions or references")
            return {
                'bleu': 0.0,
                'chrf': 0.0,
                'bertscore': 0.0,
                'valid': False
            }
        
        try:
            # BLEU
            bleu_result = self.metrics['bleu'].compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            results['bleu'] = bleu_result['bleu']
            results['bleu_precisions'] = bleu_result.get('precisions', [])
            
            # ChrF
            chrf_result = self.metrics['chrf'].compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            results['chrf'] = chrf_result['score']
            results['chrf_char_order'] = chrf_result.get('char_order', 6)
            
            # BERTScore (if available)
            if self.metrics.get('bertscore'):
                try:
                    bert_result = self.metrics['bertscore'].compute(
                        predictions=predictions,
                        references=references,
                        lang=self._get_lang_code_for_bertscore(language_pair)
                    )
                    results['bertscore'] = np.mean(bert_result['f1'])
                except Exception as e:
                    logger.warning(f"BERTScore computation failed: {e}")
                    results['bertscore'] = None
            
            results['valid'] = True
            
        except Exception as e:
            logger.error(f"Metric computation failed: {e}")
            results['valid'] = False
        
        return results
    
    def _get_lang_code_for_bertscore(self, language_pair: str) -> str:
        """Convert language pair to BERTScore language code"""
        # BERTScore uses ISO 639-1 codes
        lang_map = {
            'kik-eng': 'en',
            'eng-kik': 'en',
            'en-kik': 'en',
            'kik-en': 'en',
            'default': 'en'
        }
        return lang_map.get(language_pair, 'multi')
    
    def compute_metrics_batch(
        self,
        predictions_list: List[List[str]],
        references_list: List[List[str]],
        model_names: List[str],
        language_pair: str = "kik-eng"
    ) -> Dict[str, Dict]:
        """
        Compute metrics for multiple model outputs in parallel
        
        Args:
            predictions_list: List of prediction lists
            references_list: List of reference lists
            model_names: Names of models
            language_pair: Language pair for special tokenization
        
        Returns:
            Dict mapping model_name -> metrics_dict
        """
        results = {}
        
        for model_name, predictions, references in zip(
            model_names, predictions_list, references_list
        ):
            logger.info(f"Computing metrics for {model_name}...")
            results[model_name] = self.compute_all_metrics(
                predictions, references, language_pair
            )
        
        return results
