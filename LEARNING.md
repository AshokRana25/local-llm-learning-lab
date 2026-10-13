# Learn by following one question

Start the app using [README.md](README.md), load the sample document, then ask: **How many books may a Lantern Book Club member borrow, and for how long?**

The source says **two books for 21 days**. The interesting part is how that fact reaches the answer.

## 1. LLM inference: producing the next token

A large language model has learned numerical weights during training. During **inference**, it receives input tokens, computes probabilities for the next token, selects one, appends it, and repeats. A token is a piece of text, not necessarily a whole word. Using a model does not normally change its weights.

Ollama loads the `llama3.2:3b` model and serves it through a local HTTP API. In this lab, `src/ollama_client.py` sends a list of messages to `/api/chat`. Each message has a role and content. The response arrives in small JSON events, and Streamlit displays the text as it arrives. Streaming improves how soon you see progress; it does not remove the generation work.

The **3B** refers approximately to three billion model parameters. Quantization stores weights with fewer bits to reduce disk and memory needs, sometimes at a quality cost. The model download size is not the total RAM requirement: working buffers and the token cache also use memory.

The **context window** is the model's working space for the prompt and generated response. This app requests 4,096 tokens and at most 512 output tokens. Raising the context limit can increase memory use. Temperature affects how predictable token selection is: lower values tend to give more consistent output. Even temperature zero does not make an answer true.

Plain-chat messages are resent on later requests within the app's bounded history. This is conversational context, not learning or training. Clearing the chat removes that context.

## 2. Embeddings: text represented as numbers

An embedding model maps text into a vector: a list of numbers that captures patterns useful for similarity. "When does the club gather?" and "The club meets on Wednesday" may be close even though they use different words. Exact keyword overlap is not required.

This lab uses a separate, smaller model, `sentence-transformers/all-MiniLM-L6-v2`, which produces 384-dimensional vectors. It is an embedding model, not the chat model. It runs on CPU, and its output is normalized to unit length. Questions and documents must be embedded with the **same model and preprocessing**. Mixing vectors from different models can make retrieval meaningless even if the vector dimensions match. See the [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for its intended use and input limit.

Long documents are split into overlapping chunks because an embedding of a huge document would blur individual topics and exceed the embedding model's input limit. This project uses tokenizer offsets to preserve the original wording, with up to 200 tokens per chunk and 40 tokens of overlap. PDF chunks stay within each page so citations retain page information. Tokens here belong to the embedding tokenizer; Llama uses a different tokenizer.

Overlap gives facts near a boundary a chance to appear together. Too little overlap can separate a question's answer from its context. Too much overlap duplicates evidence and increases storage and work.

## 3. Vector database: finding nearby chunks

ChromaDB stores each chunk's embedding alongside its text and metadata: source filename, page, document identifier, and chunk number. The persistent client writes to `data/chroma/`; restarting the app reopens the same collection. A separate database server is unnecessary.

To search, the app embeds the question and asks Chroma for the nearest vectors using cosine distance. In this setup:

```text
cosine distance = 1 - cosine similarity
```

Smaller distance means a closer direction in embedding space. It does **not** mean a calibrated probability that the text is correct, relevant, or sufficient. A nearest-neighbor query can always return something when the library is nonempty, even when none of the documents answers the question. Chroma uses an approximate index, so nearest-neighbor results can also differ from an exhaustive exact search.

Persistence means you do not recompute every document embedding on every question. You still compute a query embedding. Removing an indexed document deletes its records; it does not retrain or alter the LLM.

## 4. RAG: retrieving evidence before generating

**Retrieval-augmented generation** connects the database search to the chat model:

```text
INDEX ONCE
PDF/TXT -> extract text -> overlapping chunks -> embeddings
                                                   |
                                                   v
                                      persistent Chroma collection

ASK EACH TIME
question -> query embedding -> nearest chunks from Chroma
                                      |
                                      v
           question + labeled excerpts + grounding instructions
                                      |
                                      v
                         Ollama / llama3.2:3b
                                      |
                                      v
                        answer with [1], [2] citations
```

`RAGStore.add_document()` builds the index. `RAGStore.retrieve()` searches it. `build_messages()` labels excerpts and puts them in the prompt. The model is asked to use that evidence and acknowledge missing information. `app.py` shows the exact excerpt text supplied to the model, including any shortening needed for the prompt budget.

For the book-club question, retrieval should find the lending policy, which states the two-book limit and 21-day loan period. The LLM then turns the retrieved facts into a natural-language response. It is not reading the entire document on every turn, searching the web, or absorbing the file into its weights.

This simple lab deliberately uses standalone questions for RAG. It does not rewrite follow-up questions from conversation history, rerank search results, apply a relevance threshold, or automatically validate citations. Those are useful extensions once the basic flow is clear.

There are two separate failure points:

1. **Retrieval failure:** the needed fact never reaches the prompt. Inspect chunking, extraction quality, wording, and the number of retrieved chunks.
2. **Generation failure:** the right fact is present but the model ignores, misstates, or invents information. Inspect the prompt, conflicting excerpts, and model quality.

An instruction such as "cite your sources" helps but is not a proof mechanism. Check that each citation actually supports the claim. Treat text inside documents as untrusted data; a PDF can contain instructions trying to redirect the model. The prompt discourages following them, but small models can still be fooled.

## 5. RAG versus fine-tuning

| Question | RAG | Fine-tuning |
| --- | --- | --- |
| What changes? | A document index and the inference prompt | Model weights or added trainable adapters |
| Common purpose | Answer using external, changing documents | Adapt behavior, style, or performance on a task |
| Updating a policy | Replace/remove old indexed content and re-index | Prepare training examples and train again; factual recall is still uncertain |
| Source traceability | Retrieved passages can be displayed and cited | No inherent link from generated claims to training examples |
| Hardware burden | Embedding/indexing plus normal inference | Additional training memory and compute |
| Does it guarantee truth? | No | No |

Fine-tuning is training, usually using examples of desired inputs and outputs. It is not the same as uploading a PDF or saving a chat. Some small adapter training is possible with suitable hardware, but this 16 GB beginner lab performs **no fine-tuning**. Start with RAG when the task is answering questions from a changing document collection. Fine-tuning and RAG can be combined later: one shapes behavior, while the other supplies evidence at inference time.

## 6. Experiments with observable outcomes

Change one variable at a time. Keep a table with the question, expected answer, retrieved source, actual answer, response time, and whether the citation supports the answer.

| Experiment | What to do | What to observe |
| --- | --- | --- |
| Chat versus RAG | Ask about the fictional club in Chat, then in document mode | Does the answer become supported by a visible passage? |
| Missing information | Ask for the library director's name | Does the model admit the answer is absent? Check the excerpt yourself. |
| Semantic search | Ask "When do members get together?" | Can retrieval find the meeting schedule without exact wording? |
| Retrieval size | Compare 1, 3, and 5 retrieved chunks | Does extra evidence help, or introduce irrelevant context? |
| Temperature | Compare 0.0, 0.2, and 0.8 with the same question | How much do wording, consistency, and unsupported claims change? |
| Model size | Pull `llama3.2:1b`, refresh models, and compare with 3B | Record speed, resource use, and answer quality. |
| Persistence | Index once, stop the app, restart, ask again | Are document counts and answers available without indexing again? |
| Duplicate detection | Upload the same file twice | The chunk count should not grow. |
| Conflicting versions | Make a renamed sample with a 14-day loan period | Which passages are retrieved? Remove the obsolete version and retry. |
| PDF extraction | Compare a simple text PDF with a scanned PDF | The scan needs OCR; inspect any warning instead of assuming ingestion succeeded. |
| Chunk boundaries | Try a smaller `CHUNK_TOKENS` with smaller overlap in `src/rag.py` | Rebuild the library and inspect whether facts stay together. |

Before changing chunk settings or embedding models, stop the app and rename `data/chroma` to a backup folder, then restart and index again. Otherwise old and new chunks can coexist. When changing the embedding model, also ensure it supports fast tokenizer offsets and update the collection's model metadata through a new database.

For a small evaluation, write ten answerable questions and five deliberately unanswerable ones. Score (a) whether the right passage was retrieved, (b) whether the answer is supported, (c) citation correctness, and (d) correct refusal when facts are absent. Measure more than one response: a single fluent answer is weak evidence.

## 7. Where to read the code next

1. `src/ollama_client.py`: understand a request, a streamed event, and a connection error.
2. `src/rag.py` → `extract_pages()` and `chunk_text()`: follow text into chunks.
3. `RAGStore._embed()` and `add_document()`: follow chunks into vectors and storage.
4. `retrieve()` and `build_messages()`: inspect how evidence becomes a prompt.
5. `app.py`: see how Streamlit reruns the interface while caching the store and keeping session history.
6. `tests/test_lab.py`: see how deterministic vectors and simulated model responses test the application without downloading model weights.

After the basic lab works, useful next projects include metadata filters, a query-rewriting step for follow-ups, a reranker, a relevance threshold calibrated on your evaluation questions, and a stricter token budget using the chat model's tokenizer. Add one feature at a time and keep inspecting the retrieved evidence.
