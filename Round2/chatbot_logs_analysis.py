import os
import json
import argparse
import logging
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # for prettier plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_logs(path: str) -> List[Dict[str, Any]]:
    """Load logs from a JSON file."""
    try:
        with open(path, "r") as f:
            logs = json.load(f)
        logging.info(f"Loaded {len(logs)} log entries from {path}")
        if len(logs) > 0:
            logging.info(f"Sample entry keys: {list(logs[0].keys())}")
        return logs
    except Exception as e:
        logging.error(f"Failed to load logs from {path}: {e}")
        raise


def preprocess_logs(logs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert raw log entries into a structured DataFrame."""
    rows = []
    for entry in logs:
        latency = entry.get("response_latency_ms")
        if latency is None:
            continue
        retrieval = entry.get("retrieval_time_ms", 0)
        generation = entry.get("generation_time_ms", 0)
        gen_input_tokens = entry.get("generation_input_tokens", 0)
        gen_output_tokens = entry.get("generation_output_tokens", 0)

        feedback = entry.get("user_feedback", "").lower()
        if feedback == "thumb_up":
            correct = True
        elif feedback == "thumb_down":
            correct = False
        else:
            correct = None

        chunk_sources = [c.get("source", "") for c in entry.get("retrieved_chunks", [])]
        wiki_chunks = sum(1 for s in chunk_sources if "Wiki" in s)
        pdf_chunks = sum(1 for s in chunk_sources if "PDF" in s)
        conf_chunks = sum(1 for s in chunk_sources if "Confluence" in s)

        rows.append({
            "latency_ms": latency,
            "retrieval_time_ms": retrieval,
            "generation_time_ms": generation,
            "generation_input_tokens": gen_input_tokens,
            "generation_output_tokens": gen_output_tokens,
            "answer_correct": correct,
            "wiki_chunks": wiki_chunks,
            "pdf_chunks": pdf_chunks,
            "conf_chunks": conf_chunks
        })

    df = pd.DataFrame(rows)
    numeric_cols = [
        "latency_ms", "retrieval_time_ms", "generation_time_ms",
        "generation_input_tokens", "generation_output_tokens"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=["latency_ms"])
    logging.info(f"Processed DataFrame shape: {df.shape}")
    return df


def plot_latency_distribution(df: pd.DataFrame, save_path: str) -> None:
    """Plot histogram and boxplot of latencies."""
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df["latency_ms"], bins=30, kde=True, color="skyblue")
    plt.title("Latency Distribution (ms)")
    plt.xlabel("Latency (ms)")

    plt.subplot(1, 2, 2)
    sns.boxplot(x="answer_correct", y="latency_ms", data=df, palette="Set2")
    plt.title("Latency by Answer Correctness")
    plt.xlabel("Answer Correct")
    plt.ylabel("Latency (ms)")

    plt.tight_layout()
    plt.savefig(save_path)
    logging.info(f"Latency distribution plots saved to {save_path}")
    print(f"\n[+] Latency distribution plots saved: {save_path}")


def plot_chunk_sources(df: pd.DataFrame, save_path: str) -> None:
    """Plot average chunk counts by correctness."""
    means = df.groupby("answer_correct")[["wiki_chunks", "pdf_chunks", "conf_chunks"]].mean().reset_index()
    means["answer_correct"] = means["answer_correct"].astype(str)  # convert bool to str for plotting

    means = means.melt(id_vars="answer_correct", var_name="source", value_name="avg_chunks")

    plt.figure(figsize=(8,6))
    sns.barplot(x="source", y="avg_chunks", hue="answer_correct", data=means, palette="pastel")
    plt.title("Average Number of Chunks by Source & Answer Correctness")
    plt.xlabel("Data Source")
    plt.ylabel("Average Chunk Count")
    plt.legend(title="Answer Correct")
    plt.savefig(save_path)
    logging.info(f"Chunk source bar plot saved to {save_path}")
    print(f"[+] Chunk source bar plot saved: {save_path}")


def generate_summary(df: pd.DataFrame) -> None:
    """Print a neat summary report."""
    p99_latency = np.percentile(df["latency_ms"], 99)
    avg_gen_tokens = df["generation_input_tokens"].mean()
    accuracy = df["answer_correct"].mean() if df["answer_correct"].notna().any() else None

    print("\n==== Chatbot Performance Summary ====")
    print(f"Total log entries analyzed: {len(df)}")
    print(f"P99 Latency: {p99_latency:.1f} ms")
    print(f"Average generation input tokens: {avg_gen_tokens:.1f}")
    if accuracy is not None:
        print(f"Answer correctness rate: {accuracy:.2%}")
    else:
        print("No answer correctness feedback available.")

    incorrect = df[df["answer_correct"] == False]
    correct = df[df["answer_correct"] == True]

    print("\nAverage chunk counts for incorrect answers:")
    print(incorrect[["wiki_chunks", "pdf_chunks", "conf_chunks"]].mean().to_dict())

    print("\nAverage chunk counts for correct answers:")
    print(correct[["wiki_chunks", "pdf_chunks", "conf_chunks"]].mean().to_dict())

    logging.info(f"P99 latency: {p99_latency:.1f} ms")
    logging.info(f"Average generation tokens: {avg_gen_tokens:.1f}")
    if accuracy is not None:
        logging.info(f"Answer correctness rate: {accuracy:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Analyze chatbot logs and generate reports.")
    parser.add_argument("--logfile", type=str, default="logs.json",
                        help="Path to logs.json file (default: logs.json in script directory)")
    args = parser.parse_args()

    logfile = args.logfile
    if not os.path.isabs(logfile):
        logfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), logfile)

    logs = load_logs(logfile)
    df = preprocess_logs(logs)

    generate_summary(df)

    plot_latency_distribution(df, os.path.join(os.path.dirname(logfile), "latency_distribution.png"))
    plot_chunk_sources(df, os.path.join(os.path.dirname(logfile), "chunk_source_counts.png"))

    # Quantitative Trade-off Analysis
    extra_chunks = 6
    tokens_per_chunk = 400
    queries_per_month = 100_000
    llama_cost_per_million_tokens = 3.00  # $

    extra_tokens_per_query = extra_chunks * tokens_per_chunk
    extra_tokens_per_month = extra_tokens_per_query * queries_per_month
    extra_tokens_million = extra_tokens_per_month / 1_000_000
    extra_cost_option_b = extra_tokens_million * llama_cost_per_million_tokens

    print(f"\nOption B estimated monthly cost increase: ${extra_cost_option_b:.2f}")
    logging.info(f"Option B estimated monthly cost increase: ${extra_cost_option_b:.2f}")

    p99_latency = np.percentile(df["latency_ms"], 99)
    print(f"Current P99 latency: {p99_latency:.1f} ms")
    logging.info(f"Current P99 latency: {p99_latency:.1f} ms")

    print("Option A adds 600ms fixed latency.")
    print("Option B adds 250ms retrieval latency.")

    csv_path = os.path.join(os.path.dirname(logfile), "log_analysis_summary.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Summary CSV saved as {csv_path}")
    print(f"\n[+] Summary CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
