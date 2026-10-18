"""
Diagnostic script to trace every retrieval stage for development benchmark queries:
- Query analysis outputs
- Dense candidate pool & scores
- Threshold filtering loss
- Deduplication loss
- Reranking score & ordering
- Expected message ID tracking across all stages
"""

import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from rag.config import settings
from rag.embedder import embed_query
from rag.query_analyzer import analyze_query
from rag.reranker import rerank_candidates
from rag.pipeline import RAGPipeline
from evaluation.benchmark import load_benchmark

def diagnose():
    dataset = load_benchmark(split="dev")
    pipeline = RAGPipeline()
    pipeline.load_index()

    print(f"Loaded {len(dataset.queries)} dev queries.")
    print("=" * 80)

    for q in dataset.queries:
        print(f"\n--- [{q.id}] ({q.category}) '{q.query}' ---")
        expected_ids = set(q.expected_message_ids)
        all_relevant = q.all_relevant_ids
        print(f"Target IDs: {all_relevant}")
        if q.is_negative_query:
            print("Negative Query: Expect NO documents.")

        # Stage 1: Query Analysis
        analysis = analyze_query(q.query)
        print(f"Query Analysis: type={analysis.query_type}, entities={analysis.entities}, terms={analysis.terms}, temporal={analysis.temporal_intent}")

        # Stage 2: Initial Dense Search
        initial_k = max(settings.INITIAL_RETRIEVAL_K, 20 * 3) # default 60
        query_emb = embed_query(q.query)
        dense_raw = pipeline.store.search(query_emb, top_k=initial_k)
        dense_mids = [r["message_id"] for r in dense_raw]
        found_in_dense = set(dense_mids) & all_relevant
        print(f"Stage 1 Dense Raw ({len(dense_raw)} chunks, {len(set(dense_mids))} unique emails):")
        print(f"   Relevant chunks found: {len(found_in_dense)} / {len(all_relevant)} -> {found_in_dense}")
        if all_relevant - found_in_dense:
            print(f"   LOST IN DENSE: {all_relevant - found_in_dense}")

        # Stage 3: Similarity Threshold
        threshold = settings.SIMILARITY_THRESHOLD
        filtered = [r for r in dense_raw if r["score"] >= threshold]
        filtered_mids = [r["message_id"] for r in filtered]
        found_after_thresh = set(filtered_mids) & all_relevant
        print(f"Stage 2 Threshold ({threshold:.2f}) -> {len(filtered)} chunks:")
        print(f"   Relevant remaining: {len(found_after_thresh)} / {len(all_relevant)}")
        lost_thresh = found_in_dense - found_after_thresh
        if lost_thresh:
            print(f"   LOST BY SIMILARITY THRESHOLD: {lost_thresh}")

        # Stage 4: Reranking & Min Rerank Score
        ranked = rerank_candidates(
            query=q.query,
            analysis=analysis,
            candidates=filtered,
            final_k=20,
            deduplicate=True,
        )
        ranked_mids = [r["message_id"] for r in ranked]
        ranked_mids_10 = ranked_mids[:10]
        found_at_10 = set(ranked_mids_10) & all_relevant
        print(f"Stage 3 Reranked & Deduped (Top 10):")
        print(f"   Relevant at @10: {len(found_at_10)} / {len(all_relevant)} -> {found_at_10}")
        lost_rerank = found_after_thresh - set(ranked_mids)
        if lost_rerank:
            print(f"   LOST IN RERANK/MIN_SCORE: {lost_rerank}")
        pushed_below_10 = (set(ranked_mids) & all_relevant) - found_at_10
        if pushed_below_10:
            print(f"   PUSHED BELOW TOP-10: {pushed_below_10}")

        for idx, r in enumerate(ranked[:5], 1):
            is_rel = "✓" if r["message_id"] in all_relevant else "✗"
            print(f"   [{idx}] {is_rel} {r['message_id']} | score={r.get('final_score', r.get('score', 0)):.4f} (dense={r['score']:.4f}) | {r.get('subject', '')[:50]}")

if __name__ == "__main__":
    diagnose()
