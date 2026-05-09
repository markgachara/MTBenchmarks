"""
Results aggregation and reporting.
Generates outputs matching paper Tables 4, 5, and 6 format.
"""
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class MTReporter:
    """Aggregate and report benchmarking results."""

    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"benchmark_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "translations").mkdir(exist_ok=True)
        (self.run_dir / "metrics").mkdir(exist_ok=True)
        (self.run_dir / "qualitative").mkdir(exist_ok=True)

    def create_results_table(
        self,
        all_metrics: Dict[str, Dict],
        direction: str,
        human_ref_metrics: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Create a results table matching paper Table 4/5 format.

        Returns a *numeric* DataFrame: missing values are stored as
        ``None`` / ``NaN`` so that downstream pandas/R analysis can
        treat the columns as numeric. Markdown rendering applies
        formatting at output time only (see ``save_results``).

        Args:
            all_metrics: {model_display_name: metrics_dict}
            direction: e.g. "eng->kik" or "kik->eng"
            human_ref_metrics: Corpus-level baseline for human references
        """
        rows = []

        for model_name, metrics in all_metrics.items():
            row = {
                "Model": model_name,
                "Direction": direction,
                "BLEU": metrics.get("bleu"),
                "chrF++": metrics.get("chrf_pp"),
                "BERTScore F1": metrics.get("bertscore_f1"),
                "AfriCOMET-MTL": metrics.get("africomet_mtl"),
                "Perplexity": metrics.get("pred_perplexity"),
                "TTR": metrics.get("pred_ttr"),
                "Hapax Rate": metrics.get("pred_hapax_rate"),
                "Avg Sent Len": metrics.get("pred_avg_sent_len"),
                "Inference Time (s)": metrics.get("inference_time"),
                "Speed (tok/s)": metrics.get("speed"),
            }
            rows.append(row)

        # Human reference baseline row — None for metrics that are
        # undefined for the human reference (BLEU/chrF++ against itself).
        if human_ref_metrics:
            rows.append({
                "Model": "Human Reference",
                "Direction": direction,
                "BLEU": None,
                "chrF++": None,
                "BERTScore F1": None,
                "AfriCOMET-MTL": None,
                "Perplexity": human_ref_metrics.get("human_ref_perplexity"),
                "TTR": human_ref_metrics.get("human_ref_ttr"),
                "Hapax Rate": human_ref_metrics.get("human_ref_hapax_rate"),
                "Avg Sent Len": human_ref_metrics.get("human_ref_avg_sent_len"),
                "Inference Time (s)": None,
                "Speed (tok/s)": None,
            })

        return pd.DataFrame(rows)

    # Per-column formatters used only when rendering Markdown / display
    # (CSV stays numeric so pandas/R can parse the columns correctly).
    _MARKDOWN_FORMATTERS = {
        "BLEU": "{:.2f}",
        "chrF++": "{:.2f}",
        "BERTScore F1": "{:.4f}",
        "AfriCOMET-MTL": "{:.4f}",
        "Perplexity": "{:.2f}",
        "TTR": "{:.4f}",
        "Hapax Rate": "{:.4f}",
        "Avg Sent Len": "{:.2f}",
        "Inference Time (s)": "{:.2f}",
        "Speed (tok/s)": "{:.2f}",
    }

    @classmethod
    def _format_for_markdown(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with numeric columns rendered as strings,
        with ``None`` / ``NaN`` shown as ``"—"``. Used only for
        Markdown output; CSV consumers see the numeric DataFrame."""
        out = df.copy()
        for col, fmt in cls._MARKDOWN_FORMATTERS.items():
            if col not in out.columns:
                continue
            out[col] = out[col].apply(
                lambda v: "—" if v is None or (isinstance(v, float) and pd.isna(v))
                else (fmt.format(v) if isinstance(v, (int, float)) else v)
            )
        return out

    def save_results(
        self,
        df: pd.DataFrame,
        direction: str,
        fmt: str = "csv",
    ) -> Path:
        """Save results table.

        ``fmt='csv'`` writes the *numeric* DataFrame so values are
        machine-readable. ``fmt='markdown'`` applies display formatting
        first so missing values render as "—" rather than "nan".
        """
        tag = direction.replace("->", "2")
        if fmt == "csv":
            path = self.run_dir / "metrics" / f"table_{tag}.csv"
            df.to_csv(path, index=False)
        elif fmt == "markdown":
            path = self.run_dir / "metrics" / f"table_{tag}.md"
            with open(path, "w") as f:
                f.write(self._format_for_markdown(df).to_markdown(index=False))
        else:
            path = self.run_dir / "metrics" / f"table_{tag}.json"
            df.to_json(path, orient="records", indent=2, force_ascii=False)

        logger.info(f"Saved {direction} results to {path}")
        return path

    def save_full_metrics_json(self, all_results: Dict) -> Path:
        """Save the complete metrics dictionary as JSON."""
        path = self.run_dir / "metrics" / "full_metrics.json"
        # Convert numpy values
        clean = json.loads(json.dumps(all_results, default=_json_default))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved full metrics to {path}")
        return path

    def generate_qualitative_template(
        self,
        indices: List[int],
        english_texts: List[str],
        kikuyu_texts: List[str],
        model_translations: Dict[str, Dict[str, List[str]]],
    ) -> Path:
        """
        Generate Excel spreadsheet for human qualitative assessment.

        Args:
            indices: Indices of the stratified sample
            english_texts: Full English text list
            kikuyu_texts: Full Gĩkũyũ text list
            model_translations: {model_name: {direction: [translations]}}
        """
        rows = []
        criteria = [
            "Diacritic Accuracy",
            "Idiomatic Appropriateness",
            "Biblical Register Avoidance",
            "Cultural Concept Handling",
            "Addition",
            "Omission",
            "Mistranslation",
            "Grammar",
            "Overall Score",
        ]

        for idx in indices:
            for model_name, directions in model_translations.items():
                for direction, translations in directions.items():
                    if idx < len(translations):
                        if "eng->kik" in direction:
                            source = english_texts[idx]
                            reference = kikuyu_texts[idx]
                        else:
                            source = kikuyu_texts[idx]
                            reference = english_texts[idx]

                        row = {
                            "Index": idx,
                            "Direction": direction,
                            "Model": model_name,
                            "Source": source,
                            "Reference": reference,
                            "Translation": translations[idx],
                        }
                        for c in criteria:
                            row[c] = ""  # Blank for human assessor
                        row["Notes"] = ""
                        rows.append(row)

        df = pd.DataFrame(rows)
        path = self.run_dir / "qualitative" / "human_assessment_template.xlsx"
        df.to_excel(path, index=False)
        logger.info(f"Saved qualitative assessment template ({len(rows)} rows) to {path}")
        return path

    def generate_markdown_report(
        self,
        eng2kik_df: pd.DataFrame,
        kik2eng_df: pd.DataFrame,
        hardware_info: Dict,
    ) -> Path:
        """Generate full markdown report."""
        lines = [
            f"# Gĩkũyũ MT Benchmark Report — {self.timestamp}\n",
            "## Hardware\n",
            f"- CPU: {hardware_info['cpu']['model']} ({hardware_info['cpu']['cores']} cores)",
            f"- RAM: {hardware_info['ram']['total_gb']:.1f} GB",
        ]
        if hardware_info["gpu"]["available"]:
            for g in hardware_info["gpu"]["devices"]:
                lines.append(f"- GPU {g['index']}: {g['name']} ({g['memory_gb']:.1f} GB)")
        lines.append("")

        lines.append("## Table 4: English → Gĩkũyũ\n")
        lines.append(self._format_for_markdown(eng2kik_df).to_markdown(index=False))
        lines.append("")

        lines.append("## Table 5: Gĩkũyũ → English\n")
        lines.append(self._format_for_markdown(kik2eng_df).to_markdown(index=False))
        lines.append("")

        path = self.run_dir / "report.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Saved report to {path}")
        return path


def _json_default(obj):
    """JSON serializer for non-serializable types."""
    import numpy as np
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
