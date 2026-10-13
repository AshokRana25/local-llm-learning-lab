# Your practice documents

Start with `example.txt`, a fictional book-club guide. The app's **Load the sample document** button indexes it directly.

For your own files, use the app's upload control and click **Index uploaded files**. Merely copying files into this folder does not index them.

- Supported: UTF-8 `.txt` and text-based `.pdf` files.
- Limits: 10 MB per file, 200 PDF pages, one million extracted characters per file, and 10,000 chunks across the library.
- Scans and images need OCR first. Password-protected PDFs are rejected.
- Blank PDF pages are skipped with a warning. Complex columns and tables may lose their reading order.
- The same filename and identical content are skipped on re-upload. Changed content, changed chunk settings, or a different filename creates a separate version. Remove obsolete versions under **Manage indexed documents**.

Raw uploads are read in memory. Extracted text, filenames, page numbers, and embeddings persist in `data/chroma/`. The original uploaded file is not copied here. Use **Remove selected document** to remove its indexed data. Earlier chat messages can still contain excerpts until you clear the chat or end the browser session.

Personal files in this folder are ignored by Git; `README.md` and the fictional sample are exceptions.

Try these questions in document mode:

1. On what day and at what time does the Lantern Book Club meet?
2. How many books may a member borrow, and for how long?
3. What happens if bad weather closes the library?
4. What is the name of the library director? (The expected behavior is to acknowledge missing information.)
