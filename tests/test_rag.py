import unittest
from pathlib import Path

from rag import MiniRAG


ROOT = Path(__file__).resolve().parents[1]


class RAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = MiniRAG(ROOT / "documents")

    def test_archive_never_retrieved(self):
        result = self.rag.answer("За сколько часов отменить репетиционную комнату?")
        self.assertEqual({s["doc_id"] for s in result["sources"]}, {"DOC-02"})
        self.assertIn("24", result["answer"])
        self.assertNotIn("48", result["answer"])

    def test_ambiguous_question_requests_clarification(self):
        result = self.rag.answer("За сколько часов можно отменить бронирование?")
        self.assertEqual(result["status"], "clarification_needed")

    def test_recorder_combines_two_documents(self):
        result = self.rag.answer("На какой срок можно взять портативный рекордер и куда его нужно вернуть?")
        self.assertEqual({s["doc_id"] for s in result["sources"]}, {"DOC-04", "DOC-05"})

    def test_retrieval_is_hybrid_rrf(self):
        retriever = self.rag.retriever
        self.assertTrue(hasattr(retriever, "_bm25"))
        self.assertTrue(hasattr(retriever, "_dense_index"))
        ranked = retriever.search("деньги в обеспечение возврата гитары", limit=3)
        self.assertTrue(ranked)
        self.assertTrue(all(0 < chunk.score <= 2 / (retriever.rrf_k + 1) for chunk in ranked))


if __name__ == "__main__":
    unittest.main()
