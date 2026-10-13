# Local LLM Learning Lab

A small, readable project for learning how local chat and document question answering work on a **16 GB Windows laptop**.

- **Ollama + `llama3.2:3b`** generate answers locally.
- **Streamlit** provides streaming chat, PDF/TXT uploads, and document management.
- **Sentence Transformers + `all-MiniLM-L6-v2`** embed text on your CPU.
- **ChromaDB** stores text, vectors, and source metadata on disk.
- **RAG** retrieves document excerpts and sends them with your question to the local model.

No paid API key, Docker, CUDA toolkit, or separate database server is required. Internet access is needed for installation and initial model downloads. Model weights and Python dependencies are not included in this source ZIP.

## 1. Install Python and Ollama

Use **64-bit Windows 10 22H2 or newer / Windows 11**, an x64 Intel or AMD processor, and Python **3.12** for the installation commands below. See [VALIDATION.md](VALIDATION.md) for the exact validation environments and remaining checks. Allow roughly **10–15 GB of free disk space** for Python packages, Ollama, models, caches, and a small library. This is a planning allowance, not a measured peak.

1. Install Python from [python.org's Windows downloads](https://www.python.org/downloads/windows/). With the current Python Install Manager, open a new PowerShell window and run:

   ```powershell
   py install 3.12
   py -3.12 --version
   ```

   If you use the classic installer/launcher, install a Python 3.12 **64-bit** runtime and enable the Python launcher. `py install` belongs to the new Install Manager; the version check and later `py -3.12` commands also work with the classic launcher after installing 3.12.

2. Install [Ollama for Windows](https://ollama.com/download/windows). Open it from the Start menu, then open a fresh PowerShell window:

   ```powershell
   ollama --version
   ollama pull llama3.2:3b
   ollama list
   ollama run llama3.2:3b "Explain a language model in two sentences."
   ```

The [default model download](https://ollama.com/library/llama3.2:3b) is about 2 GB. Its runtime memory usage is higher than the file size and depends on context length and hardware. The app uses a 4,096-token context and limits generated answers to 512 tokens. A GPU can speed up Ollama but is optional; CPU generation can be slow.

Ollama normally runs in the background on Windows. If it is not running, open a separate PowerShell window and run `ollama serve`. Leave that window open. If the port is already in use, Ollama is probably already running; do not start a second server.

## 2. Extract the ZIP

Right-click `local-llm-learning-lab.zip` → **Extract All**. Put the extracted project somewhere writable, preferably outside a synchronized OneDrive folder. Open the folder containing `app.py` and `requirements.txt`, click File Explorer's address bar, type `powershell`, and press Enter.

The commands below must run from that project folder. Confirm:

```powershell
Get-ChildItem
```

You should see `app.py`, `src`, `requirements.txt`, `README.md`, and `LEARNING.md`. Some extractors create an extra outer folder; enter the inner folder if needed.

## 3. Create an isolated environment and install dependencies

Copy these commands into PowerShell **one at a time** and wait for each to finish successfully:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --no-deps torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

The first package command selects CPU-only PyTorch. `--no-deps` applies only to that step; the following requirements command installs its dependencies from PyPI. Do not skip the requirements command. Embeddings run on CPU regardless of whether Ollama uses your GPU. Direct dependencies are pinned in `requirements.txt`; transitive dependencies are resolved during installation.

These commands use the environment's Python directly, so you do **not** need to activate the environment or change PowerShell's execution policy. Initial installation may take several minutes.

## 4. Start the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501) if a browser tab does not open automatically. Keep PowerShell open while using the app.

1. Click **Check Ollama / refresh models**. Leave `llama3.2:3b` selected.
2. In **Chat** mode, ask: `Explain embeddings using a library analogy.`
3. Click **Load the sample document**. First use downloads the small embedding model; this can take a few minutes.
4. Select **Ask my documents (RAG)**.
5. Ask: `On what day and at what time does the Lantern Book Club meet?`
6. The expected facts are **Wednesday, 6:30 PM**, in the **Willow Room of the Maple Community Library**. Open **View retrieved excerpts** and compare the answer with the text.
7. Upload your own PDF or TXT, then click **Index uploaded files** before asking about it.

Ask complete, standalone questions in RAG mode. Retrieval uses only the current question; it does not resolve follow-ups such as "What about its opening time?" from chat history. Plain chat remembers up to three recent complete exchanges within a small character budget. Answers from document mode are not fed into plain-chat history.

To stop Streamlit, press **Ctrl+C** in its PowerShell window. To free the LLM's memory immediately:

```powershell
ollama stop llama3.2:3b
```

Otherwise the app asks Ollama to unload the model after two idle minutes. To run the lab another day, open Ollama, open PowerShell in the project folder, and repeat the Streamlit command. Do not recreate the environment or re-index unchanged documents.

## What persists, and what stays local

| Item | Location / lifetime |
| --- | --- |
| Extracted text, vectors, source names and pages | `data/chroma/`, persists across restarts |
| Sentence Transformers model | `data/models/`, downloaded on first indexing/retrieval |
| Llama model | Ollama-managed model storage, outside this project |
| Raw uploaded PDF/TXT | Read in memory; the app does not save a copy |
| Chat messages and displayed sources | Streamlit browser session; clear with **Clear chat history** |

The app sends prompts only to `127.0.0.1:11434` and has no cloud-model fallback. Use the documented local model. Streamlit listens on loopback, usage statistics are disabled, and Chroma/Hugging Face telemetry is disabled by this project. Package and model downloads still contact their download services. Indexed text is stored unencrypted on your disk. This is a single-user local learning app; keep it bound to loopback.

After you have pulled the Llama model and indexed the sample once, you can run with the network disconnected. To prevent Hugging Face from checking online for cached model updates, start the app this way:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Only enable that setting after the embedding model has been fully downloaded. To allow downloads again in the same PowerShell window:

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
```

## Library behavior and limits

- Only PDF and UTF-8 TXT are supported; scans need OCR first. PDF page numbers are physical PDF page positions, which may differ from printed numbering.
- Maximum: 10 MB/file, 200 PDF pages/file, one million extracted characters/file, and 10,000 stored chunks. Start with short, simple English documents. Unusual PDFs can still require substantial memory during extraction.
- Chunks use up to 200 embedding-model tokens with 40-token overlap. The 256-token MiniLM input limit leaves space for special tokens. The default MiniLM model works best with English semantic similarity.
- The same filename + bytes + chunk settings is skipped on re-upload. A changed file or different filename becomes a separate version. Remove old versions to avoid conflicting answers.
- Retrieval returns the nearest 1–5 chunks; the app sends at most 6,000 excerpt characters. This is a heuristic budget, not an exact Llama-token count. Very dense code or multilingual text can still exceed the useful model context; try shorter questions and fewer chunks.
- There is no similarity cutoff or reranker. Nearest does not mean relevant, and cosine distance is **not** confidence. Inspect sources and verify whether they support the answer.
- Citation instructions and missing-information prompts improve grounding but cannot guarantee correctness. A small LLM can still invent answers, mishandle tables, or follow instructions embedded in a document. Document text is treated as data and the app does not give the model tools to execute commands.

## Project map

```text
local-llm-learning-lab/
├── app.py                  # Streamlit interface and session history
├── src/
│   ├── __init__.py
│   ├── ollama_client.py    # Local HTTP calls and streaming/error handling
│   └── rag.py              # Extraction, chunking, embeddings, Chroma, prompts
├── documents/
│   ├── README.md
│   └── example.txt         # Fictional sample with known answers
├── tests/
│   └── test_lab.py         # Local tests, no downloaded models needed
├── .streamlit/config.toml  # Loopback binding, upload size, telemetry setting
├── .gitignore
├── requirements.txt
├── README.md
├── LEARNING.md
└── VALIDATION.md           # What was actually checked before packaging
```

`data/` and `.venv/` appear later and are excluded from Git and the ZIP. Read [LEARNING.md](LEARNING.md) for the concepts and experiments.

## Check your installation

Run from the project folder after installing requirements:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py src tests
.\.venv\Scripts\python.exe -c "import streamlit, chromadb, sentence_transformers, torch, pypdf, requests; import app, src.ollama_client, src.rag; print('Imports OK')"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests exercise the real persistent Chroma store with deterministic test vectors and a fake embedding model. Ollama responses are simulated. They do not measure model quality or prove live generation works. The sample workflow in step 4 is your end-to-end check with real models.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `py -3.12` cannot find Python | Install the 64-bit 3.12 runtime, open a new terminal, and rerun the version check. Do not silently substitute the newest Python. |
| `ollama` is not recognized | Install Ollama and open a new PowerShell window so its PATH entry is picked up. |
| Cannot connect / HTTP 404 model not found | Open Ollama, run `ollama list`, and run `ollama pull llama3.2:3b` if absent. |
| Dependency download or certificate errors | Retry on a working connection. On a managed laptop use approved proxy/certificate settings; do not disable TLS verification. |
| PyTorch or Chroma DLL error | Confirm x64 Python 3.12. Install Microsoft's current Visual C++ x64 runtime if the error identifies a missing runtime DLL, then reinstall in a fresh environment. |
| First indexing seems slow | The embedding model is downloaded once. Keep the network connected and try the small sample first. |
| Hugging Face cache symlink warning | Windows can use a cache without symlinks; it may use more disk space. Administrator mode is not required for the lab. |
| No text in a PDF | Use a text-based PDF, or OCR a scan separately. Password-protected files are rejected. |
| Answers are slow / laptop is busy | Close other large apps, index fewer documents, retrieve 1–2 chunks, and avoid running multiple copies of the app. |
| Need a smaller model | Run `ollama pull llama3.2:1b`, refresh the model list, and select it. Compare quality with the 3B default. |
| Incorrect answers or citations | Read the retrieved excerpts. Use a more specific question, remove conflicting file versions, or lower temperature. |
| Database is locked | Close other copies of the app. Keep the project on a local, writable, non-synced folder. |
| Want to reset the document library | Stop the app and rename `data/chroma` to `data/chroma-backup` in File Explorer. Restart and re-index. The backup still contains extracted text. |
| Port 8501 is busy | Stop the previous app or run `...python.exe -m streamlit run app.py --server.port 8502`. |

## Official references

- [Ollama on Windows](https://docs.ollama.com/windows) and [chat API](https://docs.ollama.com/api/chat)
- [Llama 3.2 model and license information](https://ollama.com/library/llama3.2:3b)
- [Streamlit streaming output](https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream)
- [Sentence Transformers embeddings](https://sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html)
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Chroma collection configuration](https://docs.trychroma.com/docs/collections/configure)
- [pypdf extraction limitations](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)

The downloaded models and dependencies retain their own licenses. This ZIP contains project source and learning material, not their weights or distributions.
