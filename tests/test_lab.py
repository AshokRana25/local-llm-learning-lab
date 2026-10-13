"""Run: python -m unittest discover -s tests -v (no model downloads)."""

import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import requests
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ollama_client import OllamaError, list_models, stream_chat
from src.rag import MAX_CONTEXT_CHARS, RAGStore, build_messages, chunk_text, extract_pages


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}


class FakeEmbedder:
    """Deterministic test vectors, deliberately unrelated to real model quality."""
    tokenizer = FakeTokenizer()
    max_seq_length = 256

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            words = text.lower()
            vector = np.array([
                1.0 + words.count("book"),
                1.0 + words.count("volcano"),
                0.5,
            ])
            vectors.append(vector / np.linalg.norm(vector))
        return np.array(vectors)


def database_child(action, path):
    store = RAGStore(Path(path), embedder=FakeEmbedder())
    if action == "write":
        assert store.retrieve("book") == []
        first = store.add_document("club.txt", b"Book club members borrow two books for 21 days.")
        assert first["chunks"] == 1 and not first["skipped"]
        duplicate = store.add_document("club.txt", b"Book club members borrow two books for 21 days.")
        assert duplicate["skipped"] and store.count() == 1
        store.add_document("geology.txt", b"A volcano releases lava. A volcano may erupt.")
        hits = store.retrieve("book book book", 5)
        assert len(hits) == 2 and hits[0]["metadata"]["source"] == "club.txt"
        # A changed file with the same name is a distinct removable version.
        store.add_document("club.txt", b"Book loans now last 14 days.")
        assert len(store.documents()) == 3
        # Failure after the first write batch must remove partial new records.
        count_before = store.count()
        original_embed = store._embed
        calls = 0

        def fail_second_batch(texts):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated embedding failure")
            return original_embed(texts)

        with patch.object(store, "_embed", side_effect=fail_second_batch):
            try:
                store.add_document("partial.txt", ("book " * 7000).encode())
                raise AssertionError("Expected the simulated indexing failure")
            except RuntimeError:
                pass
        assert calls == 2 and store.count() == count_before
    elif action == "read":
        # This runs in a NEW process: no in-memory Chroma client can supply the data.
        assert store.count() == 3 and len(store.documents()) == 3
        hits = store.retrieve("volcano volcano", 1)
        assert hits[0]["metadata"]["source"] == "geology.txt"
        for document in store.documents():
            store.remove_document(document["id"])
        assert store.count() == 0 and store.retrieve("book") == []


class DocumentTests(unittest.TestCase):
    def test_utf8_and_empty_or_invalid_uploads(self):
        pages, warnings = extract_pages("notes.TXT", b"\xef\xbb\xbfHello world")
        self.assertEqual(pages, [(1, "Hello world")])
        self.assertFalse(warnings)
        for name, data in [("x.txt", b""), ("x.txt", b" \n"),
                           ("x.txt", b"\xff"), ("x.txt", b"a\x00b"),
                           ("x.csv", b"a,b"), ("x.pdf", b"invalid")]:
            with self.subTest(name=name, data=data):
                with self.assertRaises(ValueError):
                    extract_pages(name, data)

    def test_pdf_page_numbers_and_blank_warning(self):
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
        })
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 20 200 Td (The loan period is 21 days.) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
        writer.add_blank_page(width=300, height=300)
        output = io.BytesIO()
        writer.write(output)
        pages, warnings = extract_pages("policy.pdf", output.getvalue())
        self.assertEqual(pages, [(1, "The loan period is 21 days.")])
        self.assertIn("1 page", warnings[0])

    def test_scanned_pdf_rejected(self):
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=300)
        output = io.BytesIO()
        writer.write(output)
        with self.assertRaisesRegex(ValueError, "No readable text"):
            extract_pages("scan.pdf", output.getvalue())

    def test_chunking_covers_tail_and_preserves_overlap(self):
        chunks = chunk_text("one two three four five six seven", FakeTokenizer(), 4, 1)
        self.assertEqual(chunks, ["one two three four", "four five six seven"])
        self.assertEqual(chunk_text(" ", FakeTokenizer()), [])
        with self.assertRaises(ValueError):
            chunk_text("text", FakeTokenizer(), 2, 2)

    def test_real_chroma_persistence_retrieval_deduplication_and_rollback(self):
        # Process isolation also releases database file locks before Windows cleanup.
        with tempfile.TemporaryDirectory(prefix="llm-lab-test-") as folder:
            for action in ["write", "read"]:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--database-child", action, folder],
                    capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class PromptTests(unittest.TestCase):
    def test_rag_omits_history_and_reports_exact_prompt_excerpts(self):
        hits = [{"text": "x" * 7000, "metadata": {"source": "sample.txt", "page": 1},
                 "distance": 0.1}]
        messages, used = build_messages("Question?", [{"role": "user", "content": "old secret"}], hits)
        self.assertEqual(len(messages), 2)
        self.assertNotIn("old secret", str(messages))
        self.assertEqual(len(used[0]["text"]), MAX_CONTEXT_CHARS)
        self.assertIn(used[0]["text"], messages[-1]["content"])
        self.assertEqual(len(hits[0]["text"]), 7000)

    def test_chat_history_is_bounded_by_complete_pairs(self):
        history = [{"role": role, "content": f"turn-{number}"}
                   for number in range(10) for role in ["user", "assistant"]]
        messages, sources = build_messages("Hello", history)
        self.assertEqual(len(messages), 8)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[-2]["role"], "assistant")
        self.assertFalse(sources)
        with self.assertRaises(ValueError):
            build_messages("x" * 1001, [])


class OllamaTests(unittest.TestCase):
    def response_session(self, events, ok=True):
        response = MagicMock()
        response.ok = ok
        response.status_code = 404
        response.json.return_value = {"error": "model not found"}
        response.iter_lines.return_value = [json.dumps(event).encode() for event in events]
        response.__enter__.return_value = response
        session = MagicMock()
        session.__enter__.return_value = session
        session.post.return_value = response
        return session

    def test_streaming_and_laptop_defaults(self):
        session = self.response_session([
            {"message": {"content": "Hello "}, "done": False},
            {"message": {"content": "world"}, "done": True},
        ])
        with patch("src.ollama_client._session", return_value=session):
            self.assertEqual("".join(stream_chat([{"role": "user", "content": "Hi"}])), "Hello world")
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "llama3.2:3b")
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["num_predict"], 512)

    def test_incomplete_stream_and_model_errors(self):
        for events, ok in [([{ "message": {"content": "partial"}}], True),
                           ([{"error": "out of memory"}], True), ([], False)]:
            with self.subTest(events=events, ok=ok):
                with patch("src.ollama_client._session", return_value=self.response_session(events, ok)):
                    with self.assertRaises(OllamaError):
                        list(stream_chat([]))

    def test_service_connection_failure(self):
        session = self.response_session([])
        session.get.side_effect = requests.ConnectionError("offline")
        with patch("src.ollama_client._session", return_value=session):
            with self.assertRaisesRegex(OllamaError, "Cannot reach Ollama"):
                list_models()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--database-child":
        database_child(sys.argv[2], sys.argv[3])
    else:
        unittest.main()
