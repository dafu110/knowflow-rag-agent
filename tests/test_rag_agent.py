from pathlib import Path
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import threading
from tempfile import TemporaryDirectory
import unittest

from knowflow.agent import RagAgent
from knowflow.audit import AuditLogger
from knowflow.chunking import load_documents_from_path
from knowflow.evaluation import evaluate
from knowflow.models import Principal
from knowflow.server import HttpError, RateLimiter, create_handler, _safe_eval_path, _safe_upload_filename
from knowflow.sqlite_store import SQLiteKnowledgeStore
from knowflow.store import KnowledgeStore
from knowflow.providers import ReliableProvider
from knowflow.wsgi import create_application


class FakeComposer:
    name = "fake"

    def compose(self, question, evidence, citations):
        return f"LLM:{evidence[0]}\n\n依据：[{citations[0].chunk_id}]"


class CountingEmbeddingProvider:
    name = "counting_embeddings"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def status(self) -> dict[str, object]:
        return {"calls": len(self.calls)}


class RagAgentTest(unittest.TestCase):
    def build_agent(self) -> RagAgent:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = KnowledgeStore(Path(tmp.name) / "store")
        store.add_documents(load_documents_from_path(Path("sample_docs")))
        return RagAgent(store)

    def test_sales_question_has_citation(self) -> None:
        agent = self.build_agent()
        question = "\u9500\u552e\u5408\u540c\u5ba1\u6279\u9700\u8981\u54ea\u4e9b\u6750\u6599\uff1f"
        expected_term = "\u62a5\u4ef7\u5355"
        answer = agent.ask(question, Principal(user="alice", roles={"sales"}))
        self.assertTrue(answer.citations)
        self.assertTrue(any("sales_contract.md" in citation.source for citation in answer.citations))
        self.assertIn(expected_term, answer.answer)
        self.assertIn(answer.hallucination_risk, {"low", "medium"})

    def test_agent_can_use_external_evidence_composer(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = KnowledgeStore(Path(tmp.name) / "store")
        store.add_documents(load_documents_from_path(Path("sample_docs")))
        agent = RagAgent(store, composer=FakeComposer())
        answer = agent.ask("销售合同审批需要哪些材料？", Principal(user="alice", roles={"sales"}))
        self.assertTrue(answer.answer.startswith("LLM:"))
        self.assertTrue(answer.citations)

    def test_permission_filter_blocks_security_doc(self) -> None:
        agent = self.build_agent()
        question = "\u5ba2\u6237\u654f\u611f\u6570\u636e\u7684\u4e34\u65f6\u6388\u6743\u9700\u8981\u8bb0\u5f55\u54ea\u4e9b\u4fe1\u606f\uff1f"
        answer = agent.ask(question, Principal(user="alice", roles={"sales"}))
        self.assertFalse(any("security_access.md" in item["source"] for item in answer.retrieval_debug))
        self.assertFalse(answer.citations)
        self.assertEqual(answer.hallucination_risk, "high")

    def test_security_role_can_access_security_doc(self) -> None:
        agent = self.build_agent()
        question = "\u5ba2\u6237\u654f\u611f\u6570\u636e\u7684\u4e34\u65f6\u6388\u6743\u9700\u8981\u8bb0\u5f55\u54ea\u4e9b\u4fe1\u606f\uff1f"
        answer = agent.ask(question, Principal(user="ciso", roles={"security"}))
        self.assertTrue(answer.citations)
        self.assertTrue(any("security_access.md" in citation.source for citation in answer.citations))

    def test_ambiguous_follow_up_prompts_clarify(self) -> None:
        agent = self.build_agent()
        answer = agent.ask("怎么处理？", Principal(user="alice", roles={"sales"}))
        self.assertEqual(answer.answer_type, "clarify")
        self.assertTrue(answer.follow_up_questions)
        self.assertIn("needs_clarification", answer.unsupported_claims)

    def test_offline_eval_runs(self) -> None:
        agent = self.build_agent()
        result = evaluate(agent, Path("evals/rag_eval_set.jsonl"))
        self.assertEqual(result.total, 32)
        self.assertEqual(result.permission_leaks, 0)
        self.assertGreaterEqual(result.recall_at_k, 0.75)

    def test_store_delete_document_removes_chunks(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = KnowledgeStore(Path(tmp.name) / "store")
        store.add_documents(load_documents_from_path(Path("sample_docs")))
        document = store.documents()[0]
        self.assertTrue(store.delete_document(document.id))
        self.assertFalse(any(item.id == document.id for item in store.documents()))
        self.assertFalse(any(item.document_id == document.id for item in store.chunks()))

    def test_sqlite_store_persists_and_deletes_chunks(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = SQLiteKnowledgeStore(Path(tmp.name) / "knowflow.db")
        store.add_documents(load_documents_from_path(Path("sample_docs")))
        self.assertEqual(store.stats()["documents"], 4)
        reopened = SQLiteKnowledgeStore(Path(tmp.name) / "knowflow.db")
        document = reopened.documents()[0]
        self.assertTrue(reopened.delete_document(document.id))
        self.assertFalse(any(item.document_id == document.id for item in reopened.chunks()))

    def test_eval_path_is_whitelisted(self) -> None:
        self.assertEqual(_safe_eval_path("rag_eval_set.jsonl").name, "rag_eval_set.jsonl")
        with self.assertRaises(HttpError):
            _safe_eval_path("../README.md")
        with self.assertRaises(HttpError):
            _safe_eval_path("rag_eval_set.json")

    def test_upload_filename_is_restricted(self) -> None:
        self.assertEqual(_safe_upload_filename("../policy.md"), "policy.md")
        with self.assertRaises(HttpError):
            _safe_upload_filename("payload.exe")

    def test_api_token_blocks_mutations(self) -> None:
        tmp = TemporaryDirectory()
        store = KnowledgeStore(Path(tmp.name) / "store")
        store.add_documents(load_documents_from_path(Path("sample_docs")))
        handler = create_handler(store, api_token="secret", rate_limiter=RateLimiter(max_requests=100, window_seconds=1))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        port = server.server_address[1]
        body = json.dumps({"question": "销售合同审批需要哪些材料？", "user": "alice", "roles": ["sales"]}).encode("utf-8")
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("POST", "/ask", body=body, headers={"content-type": "application/json"})
        response = connection.getresponse()
        self.assertEqual(response.status, 401)
        response.read()
        connection.close()

        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(
            "POST",
            "/ask",
            body=body,
            headers={"content-type": "application/json", "x-knowflow-token": "secret"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        self.assertEqual(response.status, 200)
        self.assertTrue(payload["citations"])

    def test_auth_token_mapping_overrides_spoofed_roles(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = KnowledgeStore(Path(tmp.name) / "store")
        store.add_documents(load_documents_from_path(Path("sample_docs")))
        handler = create_handler(
            store,
            auth_tokens="sales-token:alice:sales;security-token:ciso:security",
            rate_limiter=RateLimiter(max_requests=100, window_seconds=1),
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        body = json.dumps(
            {
                "question": "客户敏感数据的临时授权需要记录哪些信息？",
                "user": "mallory",
                "roles": ["security"],
            }
        ).encode("utf-8")
        port = server.server_address[1]
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(
            "POST",
            "/ask",
            body=body,
            headers={"content-type": "application/json", "x-knowflow-token": "sales-token"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        self.assertEqual(response.status, 200)
        self.assertFalse(payload["citations"])

        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(
            "POST",
            "/ask",
            body=body,
            headers={"content-type": "application/json", "x-knowflow-token": "security-token"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        connection.close()
        self.assertEqual(response.status, 200)
        self.assertTrue(payload["citations"])

    def test_ask_writes_audit_summary(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = KnowledgeStore(Path(tmp.name) / "store")
        store.add_documents(load_documents_from_path(Path("sample_docs")))
        audit_path = Path(tmp.name) / "audit" / "events.jsonl"
        handler = create_handler(
            store,
            rate_limiter=RateLimiter(max_requests=100, window_seconds=1),
            audit_logger=AuditLogger(audit_path),
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        body = json.dumps({"question": "销售合同审批需要哪些材料？", "user": "alice", "roles": ["sales"]}).encode("utf-8")
        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        connection.request(
            "POST",
            "/ask",
            body=body,
            headers={"content-type": "application/json", "x-request-id": "test-request"},
        )
        response = connection.getresponse()
        response.read()
        connection.close()

        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("x-request-id"), "test-request")
        rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[-1]["action"], "ask")
        self.assertEqual(rows[-1]["request_id"], "test-request")
        self.assertEqual(rows[-1]["principal"]["user"], "alice")
        self.assertEqual(rows[-1]["citations"], 2)
        self.assertNotIn("answer", rows[-1])

    def test_retriever_reuses_document_embeddings(self) -> None:
        agent = self.build_agent()
        provider = CountingEmbeddingProvider()
        agent.embedding_provider = provider
        principal = Principal(user="alice", roles={"sales"})
        agent.ask("销售合同审批需要哪些材料？", principal)
        agent.ask("合同常见退回原因是什么？", principal)
        self.assertEqual(len(provider.calls), 3)
        self.assertGreater(len(provider.calls[0]), 1)
        self.assertEqual(len(provider.calls[1]), 1)
        self.assertEqual(len(provider.calls[2]), 1)

    def test_provider_retry_and_circuit_telemetry(self) -> None:
        provider = ReliableProvider("test_provider")
        for _ in range(3):
            with self.assertRaises(RuntimeError):
                provider._request(lambda: (_ for _ in ()).throw(RuntimeError("offline")))
        status = provider.status()
        self.assertEqual(status["failures"], 3)
        self.assertEqual(status["retries"], 3)
        self.assertEqual(status["circuit"], "open")

    def test_wsgi_serves_dashboard_static_upload_and_delete(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = KnowledgeStore(Path(tmp.name) / "store")
        app = create_application(store, auth_tokens="")

        def request(method: str, path: str, body: bytes = b"", content_type: str = "") -> tuple[str, dict[str, str], bytes]:
            captured: dict[str, object] = {}

            def start_response(status: str, headers: list[tuple[str, str]]) -> None:
                captured["status"] = status
                captured["headers"] = dict(headers)

            environ = {
                "REQUEST_METHOD": method,
                "PATH_INFO": path,
                "CONTENT_LENGTH": str(len(body)),
                "CONTENT_TYPE": content_type,
                "wsgi.input": __import__("io").BytesIO(body),
                "REMOTE_ADDR": "127.0.0.1",
            }
            payload = b"".join(app(environ, start_response))
            return str(captured["status"]), dict(captured["headers"]), payload

        status, _, html = request("GET", "/")
        self.assertTrue(status.startswith("200"))
        self.assertIn(b"KnowFlow RAG Agent", html)
        status, _, script = request("GET", "/static/app.js")
        self.assertTrue(status.startswith("200"))
        self.assertIn(b"apiFetch", script)

        boundary = "----knowflow-test"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"policy.md\"\r\n"
            "Content-Type: text/markdown\r\n\r\n# Policy\nA policy body.\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        status, _, response = request("POST", "/upload", body, f"multipart/form-data; boundary={boundary}")
        self.assertTrue(status.startswith("200"))
        document_id = json.loads(response)["document_id"]
        status, _, response = request("POST", "/ask", json.dumps({"question": "policy", "user": "alice", "roles": ["sales"]}).encode("utf-8"), "application/json")
        self.assertTrue(status.startswith("200"))
        self.assertIn("answer", json.loads(response))
        status, _, response = request("DELETE", f"/documents?id={document_id}")
        self.assertTrue(status.startswith("200"))
        self.assertTrue(json.loads(response)["deleted"])


if __name__ == "__main__":
    unittest.main()
