"""
Metric computation for MT evaluation.
Implements the full metric suite from the paper (Section 3.4):
  - MT-specific: BLEU, chrF++, BERTScore, AfriCOMET-MTL
  - Corpus-level: Perplexity, TTR, Hapax Legomena, Avg Sentence Length
"""
import logging
import math
from typing import List, Dict, Optional
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


class MTEvaluator:
    """Compute all evaluation metrics for MT benchmarking."""

    def __init__(self):
        self._sacrebleu = None
        self._bertscore = None
        self._africomet_model = None
        self._goldfish_model = None
        self._goldfish_tok = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _get_sacrebleu(self):
        if self._sacrebleu is None:
            import sacrebleu as sb
            self._sacrebleu = sb
        return self._sacrebleu

    def _get_bertscore(self):
        if self._bertscore is None:
            from bert_score import score as bert_score_fn
            self._bertscore = bert_score_fn
        return self._bertscore

    def _get_africomet(self):
        if self._africomet_model is None:
            try:
                from comet import download_model, load_from_checkpoint
                model_path = download_model("masakhane/africomet-mtl")
                self._africomet_model = load_from_checkpoint(model_path)
                logger.info("Loaded AfriCOMET-MTL model")
            except Exception as e:
                logger.warning(f"Could not load AfriCOMET-MTL: {e}")
                self._africomet_model = None
        return self._africomet_model

    def _get_goldfish(self):
        """Load Goldfish-Kikuyu (124M GPT-2) for perplexity scoring."""
        if self._goldfish_model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model_id = "goldfish-models/kik_latn_full"
                self._goldfish_tok = AutoTokenizer.from_pretrained(model_id)
                # FP32 on a single GPU; the model is only ~500 MB.
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._goldfish_model = (
                    AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
                )
                logger.info(f"Loaded Goldfish-Kikuyu LM on {device}")
            except Exception as e:
                logger.warning(f"Could not load Goldfish-Kikuyu: {e}")
                self._goldfish_model = None
                self._goldfish_tok = None
        return self._goldfish_model, self._goldfish_tok

    def release_gpu(self):
        """Release any GPU-resident models (AfriCOMET, BERTScore cache, Goldfish) to free VRAM."""
        import gc
        import torch
        if self._africomet_model is not None:
            del self._africomet_model
            self._africomet_model = None
        if self._goldfish_model is not None:
            del self._goldfish_model
            self._goldfish_model = None
            self._goldfish_tok = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Evaluator GPU memory released")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
        sources: Optional[List[str]] = None,
        direction: str = "kik->eng",
    ) -> Dict:
        """
        Compute all available metrics.

        Args:
            predictions: Model translations
            references: Human reference translations
            sources: Source texts (needed for AfriCOMET)
            direction: Translation direction string
        """
        if not predictions or not references:
            logger.warning("Empty predictions or references")
            return {"valid": False}

        results = {"valid": True, "num_sentences": len(predictions)}

        # MT-specific metrics
        results.update(self._compute_bleu(predictions, references))
        results.update(self._compute_chrf_pp(predictions, references))
        results.update(self._compute_bertscore(predictions, references, direction))
        if sources:
            results.update(
                self._compute_africomet(predictions, references, sources)
            )

        # Corpus-level linguistic metrics
        results.update(
            self._compute_corpus_metrics(predictions, references, direction)
        )

        # Perplexity (only meaningful for Kikuyu output, scored by a Kikuyu LM)
        if direction.endswith("kik"):
            ppl = self._compute_perplexity(predictions)
            if ppl is not None:
                results["pred_perplexity"] = ppl

        return results

    # ------------------------------------------------------------------
    # MT-Specific Metrics
    # ------------------------------------------------------------------

    def _compute_bleu(self, predictions: List[str], references: List[str]) -> Dict:
        """BLEU via SacreBLEU (standardized tokenization)."""
        try:
            sb = self._get_sacrebleu()
            bleu = sb.corpus_bleu(predictions, [references])
            return {
                "bleu": bleu.score,
                "bleu_precisions": list(bleu.precisions),
                "bleu_bp": bleu.bp,
            }
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            return {"bleu": None}

    def _compute_chrf_pp(self, predictions: List[str], references: List[str]) -> Dict:
        """chrF++ via SacreBLEU (char_order=6, word_order=2)."""
        try:
            sb = self._get_sacrebleu()
            chrf = sb.corpus_chrf(predictions, [references], word_order=2)
            return {"chrf_pp": chrf.score}
        except Exception as e:
            logger.warning(f"chrF++ computation failed: {e}")
            return {"chrf_pp": None}

    def _compute_bertscore(
        self, predictions: List[str], references: List[str], direction: str
    ) -> Dict:
        """BERTScore F1 using bert-base-multilingual-cased."""
        try:
            bert_score_fn = self._get_bertscore()
            # Use the target language side for BERTScore
            lang = "en" if "eng" in direction.split("->")[-1] else "en"
            P, R, F1 = bert_score_fn(
                predictions,
                references,
                model_type="bert-base-multilingual-cased",
                lang=lang,
                verbose=False,
            )
            return {
                "bertscore_f1": float(F1.mean()),
                "bertscore_precision": float(P.mean()),
                "bertscore_recall": float(R.mean()),
            }
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
            return {"bertscore_f1": None}

    def _compute_africomet(
        self,
        predictions: List[str],
        references: List[str],
        sources: List[str],
    ) -> Dict:
        """AfriCOMET-MTL: primary ranking metric."""
        model = self._get_africomet()
        if model is None:
            return {"africomet_mtl": None}

        try:
            data = [
                {"src": src, "mt": mt, "ref": ref}
                for src, mt, ref in zip(sources, predictions, references)
            ]
            output = model.predict(data, batch_size=8, gpus=1)
            scores = output.scores if hasattr(output, "scores") else output[0]
            return {
                "africomet_mtl": float(np.mean(scores)),
                "africomet_mtl_scores": [float(s) for s in scores],
            }
        except Exception as e:
            logger.warning(f"AfriCOMET-MTL computation failed: {e}")
            return {"africomet_mtl": None}

    # ------------------------------------------------------------------
    # Corpus-Level Linguistic Metrics
    # ------------------------------------------------------------------

    def _compute_corpus_metrics(
        self,
        predictions: List[str],
        references: List[str],
        direction: str,
    ) -> Dict:
        """Compute corpus-level linguistic metrics for both predictions and references."""
        results = {}

        # Metrics on predictions
        pred_metrics = self._linguistic_metrics(predictions, prefix="pred")
        results.update(pred_metrics)

        # Metrics on references (baseline)
        ref_metrics = self._linguistic_metrics(references, prefix="ref")
        results.update(ref_metrics)

        return results

    @staticmethod
    def _linguistic_metrics(texts: List[str], prefix: str) -> Dict:
        """Compute TTR, Hapax Legomena, avg sentence length for a corpus."""
        all_tokens = []
        sent_lengths = []

        for text in texts:
            tokens = text.strip().split()
            all_tokens.extend(tokens)
            sent_lengths.append(len(tokens))

        if not all_tokens:
            return {}

        token_counts = Counter(all_tokens)
        total_tokens = len(all_tokens)
        unique_tokens = len(token_counts)

        ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
        hapax = sum(1 for c in token_counts.values() if c == 1)
        hapax_rate = hapax / unique_tokens if unique_tokens > 0 else 0

        return {
            f"{prefix}_ttr": round(ttr, 4),
            f"{prefix}_hapax_rate": round(hapax_rate, 4),
            f"{prefix}_avg_sent_len": round(float(np.mean(sent_lengths)), 2),
            f"{prefix}_std_sent_len": round(float(np.std(sent_lengths)), 2),
            f"{prefix}_total_tokens": total_tokens,
            f"{prefix}_unique_tokens": unique_tokens,
        }

    # ------------------------------------------------------------------
    # Perplexity (Goldfish-Kikuyu LM)
    # ------------------------------------------------------------------

    def _compute_perplexity(self, texts: List[str]) -> Optional[float]:
        """Corpus-level perplexity scored by Goldfish-Kikuyu (124M GPT-2).

        Returns exp(corpus mean per-token NLL), summing total NLL across
        sentences and dividing by total token count (paper-standard
        formulation). Returns None on any failure or if the model is
        unavailable.
        """
        model, tok = self._get_goldfish()
        if model is None or tok is None:
            return None
        try:
            import torch
            device = next(model.parameters()).device
            total_nll = 0.0
            total_tokens = 0
            max_len = getattr(model.config, "max_position_embeddings", 512) or 512
            with torch.no_grad():
                for text in texts:
                    if not text or not text.strip():
                        continue
                    enc = tok(
                        text.strip(),
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_len,
                    )
                    ids = enc.input_ids.to(device)
                    if ids.shape[1] < 2:  # need at least 2 tokens for next-token loss
                        continue
                    out = model(ids, labels=ids)
                    n_tokens = ids.shape[1] - 1  # next-token prediction
                    total_nll += out.loss.item() * n_tokens
                    total_tokens += n_tokens
            if total_tokens == 0:
                return None
            return float(math.exp(total_nll / total_tokens))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Perplexity computation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Human Reference Baseline
    # ------------------------------------------------------------------

    def compute_reference_baseline(
        self, references: List[str], direction: str
    ) -> Dict:
        """
        Compute corpus-level metrics for human reference translations.
        These serve as the baseline row in results Tables 4 & 5.
        """
        result = self._linguistic_metrics(references, prefix="human_ref")
        # Score the Kikuyu reference column with Goldfish-Kikuyu so the
        # Human Reference row in Table 4 has a perplexity baseline. The
        # English reference (kik->eng direction) is scored with a Kikuyu
        # LM only as a sanity check; we skip it.
        if direction.endswith("kik"):
            ppl = self._compute_perplexity(references)
            if ppl is not None:
                result["human_ref_perplexity"] = ppl
        return result
