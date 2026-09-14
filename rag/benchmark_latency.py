import time
import statistics

from rag.pipeline import RAGPipeline


QUERIES = [
    "placement interviews",
    "NPTEL courses",
    "Microsoft online assessment",
    "upcoming placement tests",
    "VISA online test",
    "placement opportunities",
    "academic examinations",
    "company recruitment tests",
    "internship opportunities",
    "important university announcements",
]


def percentile(values, p):
    values = sorted(values)
    index = int((p / 100) * (len(values) - 1))
    return values[index]


pipeline = RAGPipeline()

print("Loading FAISS index and embedding model...")
pipeline.load_index()

print("\nWarming up...")

# Warm-up runs are important because model loading/cache initialization
# should not be counted as normal user-query latency.
for query in QUERIES[:3]:
    pipeline.search(query)

print("Warm-up complete.\n")


latencies = []

for query in QUERIES:
    start = time.perf_counter()

    response = pipeline.search(query)

    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)

    print(f"{latency_ms:8.2f} ms | {query}")


print("\n" + "=" * 55)
print("MailMind Retrieval Latency Benchmark")
print("=" * 55)

print(f"Queries tested : {len(latencies)}")
print(f"Minimum        : {min(latencies):.2f} ms")
print(f"Maximum        : {max(latencies):.2f} ms")
print(f"Mean           : {statistics.mean(latencies):.2f} ms")
print(f"Median (P50)   : {statistics.median(latencies):.2f} ms")
print(f"P95            : {percentile(latencies, 95):.2f} ms")

if percentile(latencies, 95) < 200:
    print("\nRESULT: Sub-200ms P95 retrieval latency VERIFIED.")
else:
    print("\nRESULT: Sub-200ms P95 retrieval latency NOT VERIFIED.")