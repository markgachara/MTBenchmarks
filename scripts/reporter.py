"""
Results aggregation and reporting
"""
import logging
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import pandas as pd

from scripts.utils import get_hardware_info

logger = logging.getLogger(__name__)


class MTReporter:
    """Aggregate and report benchmarking results"""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def create_results_dataframe(
        self,
        results: Dict,
        hardware_info: Dict = None
    ) -> pd.DataFrame:
        """
        Convert results dict to pandas DataFrame
        
        Args:
            results: Results dict with structure:
                {model_name: {direction: {dataset: metrics_dict}}}
            hardware_info: Optional hardware info to include
        
        Returns:
            DataFrame with one row per model+direction+dataset combination
        """
        rows = []
        
        for model_name, directions in results.items():
            for direction, datasets in directions.items():
                for dataset_name, metrics in datasets.items():
                    row = {
                        'timestamp': self.timestamp,
                        'model': model_name,
                        'direction': direction,
                        'dataset': dataset_name,
                        # Metrics
                        'bleu': metrics.get('bleu', None),
                        'chrf': metrics.get('chrf', None),
                        'chrf_char_order': metrics.get('chrf_char_order', None),
                        'bertscore': metrics.get('bertscore', None),
                        # Performance
                        'speed_tokens_per_sec': metrics.get('speed', None),
                        'num_sentences': metrics.get('num_sentences', None),
                        'inference_time_sec': metrics.get('inference_time', None),
                        # Metadata
                        'valid': metrics.get('valid', False),
                        'notes': metrics.get('notes', '')
                    }
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by model, direction, dataset
        df = df.sort_values(['model', 'direction', 'dataset']).reset_index(drop=True)
        
        return df
    
    def save_csv(self, df: pd.DataFrame, filename: str = None) -> Path:
        """Save results as CSV"""
        if filename is None:
            filename = f"mt_benchmark_{self.timestamp}.csv"
        
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved results to {filepath}")
        return filepath
    
    def save_json(self, results: Dict, filename: str = None) -> Path:
        """Save detailed results as JSON"""
        if filename is None:
            filename = f"mt_benchmark_{self.timestamp}.json"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved detailed results to {filepath}")
        return filepath
    
    def generate_markdown_report(
        self,
        df: pd.DataFrame,
        hardware_info: Dict = None
    ) -> str:
        """Generate markdown summary report"""
        report = []
        report.append(f"# MT Benchmarking Report - {self.timestamp}\n")
        
        # Hardware info
        if hardware_info:
            report.append("## Hardware Configuration\n")
            report.append(f"- OS: {hardware_info['os']}")
            report.append(f"- CPU: {hardware_info['cpu']['model']} ({hardware_info['cpu']['cores']} cores)")
            report.append(f"- RAM: {hardware_info['ram']['total_gb']:.1f} GB")
            if hardware_info['gpu']['available']:
                for gpu in hardware_info['gpu']['devices']:
                    report.append(f"- GPU: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
            else:
                report.append("- GPU: Not available (CPU-only mode)")
            report.append("")
        
        # Summary statistics
        report.append("## Summary Statistics\n")
        report.append(f"- Models evaluated: {df['model'].nunique()}")
        report.append(f"- Directions: {df['direction'].nunique()}")
        report.append(f"- Datasets: {df['dataset'].nunique()}")
        report.append("")
        
        # Results table
        report.append("## Results\n")
        report.append("```")
        report.append(df.to_string(index=False))
        report.append("```\n")
        
        # Best models per metric
        report.append("## Best Performing Models\n")
        for metric in ['bleu', 'chrf', 'bertscore']:
            if metric in df.columns and df[metric].notna().any():
                best_row = df.nlargest(1, metric).iloc[0]
                score = best_row[metric]
                if pd.notna(score):
                    report.append(f"- **{metric.upper()}**: {best_row['model']} ({best_row['direction']}) = {score:.2f}")
        report.append("")
        
        # Speed comparison
        report.append("## Speed Comparison (tokens/sec)\n")
        if 'speed_tokens_per_sec' in df.columns:
            speed_df = df[df['speed_tokens_per_sec'].notna()].sort_values(
                'speed_tokens_per_sec', ascending=False
            )[['model', 'direction', 'speed_tokens_per_sec']].drop_duplicates()
            report.append(speed_df.to_markdown(index=False))
            report.append("")
        
        # Notes
        report.append("## Notes\n")
        report.append("- BLEU and ChrF scores typically range 0-100 for MT tasks")
        report.append("- Higher is better for all metrics")
        report.append("- Results depend on hardware, quantization mode, and batch size")
        report.append("- Gĩkũyũ is low-resource; expect lower absolute scores than high-resource pairs")
        
        return '\n'.join(report)
    
    def save_markdown_report(self, report: str, filename: str = None) -> Path:
        """Save markdown report"""
        if filename is None:
            filename = f"mt_benchmark_{self.timestamp}.md"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Saved markdown report to {filepath}")
        return filepath
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary to console"""
        print("\n" + "="*80)
        print(f"BENCHMARKING RESULTS - {self.timestamp}")
        print("="*80 + "\n")
        print(df.to_string(index=False))
        print("\n" + "="*80 + "\n")
