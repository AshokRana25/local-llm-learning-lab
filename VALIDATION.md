# Validation record

Prepared on **13 October 2026** on Windows x64.

## Passed

- **Python 3.12.14:** all five Python source/test files passed `compileall` and AST syntax parsing.
- **Python 3.13.13:** imported `src`, `src.ollama_client`, and `src.rag` using real installed dependencies: ChromaDB 1.5.9, Requests 2.34.2, pypdf 6.20.1, and NumPy 2.4.6.
- **10 automated tests passed** with Python 3.13.13. The full `tests/test_lab.py` suite completed successfully.
- The tests cover UTF-8 input, empty/invalid files, text extraction and page numbers in a generated PDF, a blank/scanned-style PDF, chunk boundaries and overlap, prompt limits, bounded chat history, and simulated Ollama streaming/errors.
- The database test uses **real ChromaDB persistence** and deterministic test embeddings. It checks retrieval order, unchanged-upload deduplication, distinct versions of a changed file, rollback after a failed indexing batch, and document removal.
- Persistence was verified by writing records in one process and reopening/querying them in a **new process**.
- The ZIP was checked for its required files, correct single project-root folder, readable Python sources, and archive integrity. No virtual environment, model weights, personal documents, database, or Python cache files are included.

## Not verified here

- A clean installation of **all** pinned packages did not complete in this environment. Downloads of the `hf-xet` dependency repeatedly failed with interrupted connections / Windows temporary-file locking. The CPU PyTorch package downloaded and installed, but a complete working ML environment was not established.
- Full Streamlit and Sentence Transformers imports, interactive browser behavior, the real MiniLM model download/embedding step, and actual Ollama generation were **not** tested.
- Ollama and the Llama model were not installed or downloaded as part of packaging.
- The 16 GB configuration is a conservative design choice, not a measured RAM or performance guarantee on a particular laptop.

The installation guide targets Python 3.12. Syntax was checked there; the available Python 3.13 environment allowed the real database and component tests to run without further downloads. That is not equivalent to full end-to-end validation of the recommended installation.

## Complete the live check on your laptop

Follow README.md to install requirements and pull `llama3.2:3b`. Run `pip check` and the documented import check, launch the app, load the sample, and ask the meeting-time and loan-period questions. Inspect the retrieved excerpts, then restart the app to verify your own library persists. Test again after disconnecting the network once both models are cached.

The tests simulate the embedding model and Ollama responses so they do not download models or send document content to an external service. They verify application behavior, not the factual quality of a language model.
