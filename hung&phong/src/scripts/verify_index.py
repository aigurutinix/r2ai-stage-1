"""Verify Qdrant count + retrieval sanity trên vài probe query. Temp."""
from backend.rag import RAGPipeline

rag = RAGPipeline()  # 1 client duy nhất — embedded Qdrant khoá file theo path
print("Collection:", rag.store.collection)
print("Count      :", f"{rag.store.count():,}")
print("=" * 70)

probes = [
    "Vốn điều lệ của công ty cổ phần được quy định thế nào?",
    "Doanh nghiệp nhỏ và vừa được hỗ trợ gì về thuế?",
    "Thời gian thử việc tối đa với người lao động là bao lâu?",
    "Công ty TNHH một thành viên có được giảm vốn điều lệ không?",
]
for q in probes:
    hits = rag.retrieve(q, top_k=5)
    print(f"\nQ: {q}")
    for h in hits[:5]:
        p = h.get("payload", {})
        print(f"  {h['score']:.3f} | {str(p.get('so_ky_hieu')):16s} | Điều {p.get('dieu_so')} "
              f"| {str(p.get('title'))[:40]}")
