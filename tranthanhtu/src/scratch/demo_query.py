"""Demo: in ra các điều luật retrieval lôi ra cho 1 câu hỏi bất kỳ.
Usage: python demo_query.py "câu hỏi..."
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
from local_rag_engine import LocalLegalRAGEngine

q = sys.argv[1] if len(sys.argv) > 1 else "Doanh nghiệp nhỏ và vừa được hỗ trợ gì?"
eng = LocalLegalRAGEngine(use_reranker=True)
print("\n" + "=" * 72)
print("CÂU HỎI:", q)
print("=" * 72)
for i, r in enumerate(eng.retrieve(q, top_k=5), 1):
    sc = r.get("rerank_score")
    print(f"\n[{i}] điểm liên quan={sc:.2f} | {r['doc_number']} | {r['article']}")
    print(f"    {r['clean_name']}")
    print(f"    {r['text'][:200].strip()}...")
