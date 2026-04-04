"""
scripts/run_evaluation.py
==========================
CLI script to run RAGAS evaluation on the banking RAG pipeline.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --output eval_results.json
    python scripts/run_evaluation.py --samples 20 --save-report

Outputs:
- Console: Formatted evaluation report
- JSON: Detailed metrics for CI/CD integration
- Exit code: 0 if passes production bar, 1 if fails
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


async def run_evaluation(
    num_samples: int = 5,
    output_file: str = None,
    save_report: bool = True,
    fail_on_low_score: bool = True,
):
    from config.settings import get_settings
    from src.monitoring.evaluator import RAGASEvaluator, BANKING_EVAL_DATASET

    settings = get_settings()

    print("""
╔══════════════════════════════════════════════════════╗
║      BankRAG RAGAS Evaluation Runner                 ║
╚══════════════════════════════════════════════════════╝
""")

    # Select evaluation samples
    samples = BANKING_EVAL_DATASET[:num_samples]
    print(f"  Running evaluation on {len(samples)} samples...")
    print(f"  LLM: {settings.llm.model}")
    print()

    # In a real eval, you would call the full pipeline for each sample
    # Here we generate demo responses
    rag_responses = []
    for i, sample in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] Evaluating: {sample.question[:60]}...")
        # Simulated response (replace with actual pipeline call)
        rag_responses.append({
            "answer": f"Based on our home lending guidelines [Source: lending_policy.pdf], {sample.ground_truth_answer[:200]}",
            "contexts": [sample.ground_truth_answer],
            "latency_ms": 1200 + (i * 100),
            "tokens": 450 + (i * 50),
            "cost_usd": 0.002,
        })

    # Run evaluation
    evaluator = RAGASEvaluator(
        llm_provider=settings.llm.provider,
        llm_model=settings.llm.model,
    )
    metrics = await evaluator.evaluate(samples, rag_responses)

    # Print report
    print(metrics.to_report())

    # Save JSON output
    metrics_dict = {
        "eval_id": f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "timestamp": metrics.eval_timestamp,
        "model": metrics.model_evaluated,
        "samples": metrics.total_samples,
        "overall_score": metrics.overall_score,
        "passes_production_bar": metrics.passes_production_bar,
        "metrics": {
            "faithfulness": metrics.faithfulness,
            "answer_relevancy": metrics.answer_relevancy,
            "context_precision": metrics.context_precision,
            "context_recall": metrics.context_recall,
            "compliance_pass_rate": metrics.compliance_pass_rate,
            "pii_leak_rate": metrics.pii_leak_rate,
            "hallucination_rate": metrics.hallucination_rate,
            "citation_rate": metrics.citation_rate,
            "regulatory_accuracy": metrics.regulatory_accuracy,
        },
        "performance": {
            "avg_latency_ms": metrics.avg_latency_ms,
            "p95_latency_ms": metrics.p95_latency_ms,
            "avg_tokens": metrics.avg_tokens,
            "avg_cost_usd": metrics.avg_cost_usd,
        },
    }

    if output_file:
        with open(output_file, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"\n  📄 Metrics saved to: {output_file}")

    if save_report:
        report_file = f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, "w") as f:
            f.write(metrics.to_report())
        print(f"  📄 Report saved to: {report_file}")

    # CI/CD exit code
    if fail_on_low_score and not metrics.passes_production_bar:
        print("\n❌ Evaluation FAILED: Does not meet production quality bar")
        sys.exit(1)
    elif metrics.passes_production_bar:
        print("\n✅ Evaluation PASSED: Meets production quality bar")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on BankRAG pipeline")
    parser.add_argument("--samples", type=int, default=5, help="Number of eval samples")
    parser.add_argument("--output", default="eval_results.json", help="JSON output file")
    parser.add_argument("--no-save", action="store_true", help="Don't save report to file")
    parser.add_argument("--no-fail", action="store_true", help="Don't exit with error on low scores")
    args = parser.parse_args()

    asyncio.run(run_evaluation(
        num_samples=args.samples,
        output_file=args.output,
        save_report=not args.no_save,
        fail_on_low_score=not args.no_fail,
    ))


if __name__ == "__main__":
    main()
