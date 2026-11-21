# Document Q&A Chatbot with Evaluation

A Python chatbot that answers customer questions about a grocery delivery app (Kikapu) using only its help documents, plus an automated evaluation that measures how often the answers are right.

It uses **retrieval-augmented generation (RAG)**: instead of trusting the model's memory, it finds the most relevant parts of the documents and gives them to an LLM along with the question.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/salam2345/Document-Q-A-Chatbot)

## How it works

```
question
   │
   ▼
1. Chunking     docs/*.md are split into small, labelled pieces    (qa/chunking.py)
   │
   ▼
2. Retrieval    BM25 keyword search picks the top-k pieces          (qa/retriever.py)
   │
   ▼
3. Generation   pieces + question + rules are sent to the LLM       (qa/pipeline.py, qa/llm.py)
   │
   ▼
answer + sources
```

**Evaluation** (`evaluate.py`) runs 25 test questions from `eval/questions.jsonl` and scores every answer automatically (`qa/scoring.py`):

- **22 answerable questions:** the answer must contain the expected facts, such as "$4.99" or "48 hours".
- **3 unanswerable questions:** the bot must say it doesn't know instead of inventing an answer (a hallucination).

It compares two configurations:

| | Baseline | Improved |
| --- | --- | --- |
| Chunking | whole 300-word blocks, structure ignored | split by section, 120 words with 30-word overlap, labelled with the section title |
| Chunks sent to the LLM | 1 | 4 |
| Prompt | "Answer the user's question." | answer only from the context, say "I don't know" otherwise, keep exact numbers, give both cities when they differ |

## Results

Run `python evaluate.py --config both` and paste the table it prints here.

| Config | Model | Accuracy | Correct | Retrieval hit rate | Refusal accuracy |
| --- | --- | --- | --- | --- | --- |
| baseline | | | | | |
| improved | | | | | |

## Run it in GitHub Codespaces

1. **Get a free API key.** Sign up at [Groq](https://console.groq.com/keys) and create an API key. (Any OpenAI-compatible provider works; see `.env.example` for Google Gemini.)
2. **Add the key as a Codespaces secret.** On GitHub, go to Settings → Codespaces → Secrets → New secret. Name it `LLM_API_KEY`, paste the key, and give it access to this repository. Alternatively, copy `.env.example` to `.env` inside the Codespace and paste the key there. `.env` is git-ignored, so the key never gets committed.
3. **Open the Codespace.** Click the badge above (or Code → Codespaces → Create codespace). Dependencies install automatically.
4. **Run it:**

```bash
pytest                                  # unit tests (no API key needed)
python evaluate.py --config both --provider mock   # offline dry run of the evaluation
python chat.py                          # chat in the terminal
streamlit run app.py                    # web interface (opens in a browser tab)
python evaluate.py --config both        # real evaluation with the LLM (takes ~2 minutes)
```

Detailed per-question results are saved to `results/`. Open the JSON files to see which questions failed and what the bot actually said.

## Project structure

```
docs/                 the knowledge base (sample help-center articles)
eval/questions.jsonl  25 test questions with expected keywords
qa/chunking.py        splits documents into chunks (two strategies)
qa/retriever.py       BM25 search, written out by hand
qa/llm.py             OpenAI-compatible client + offline mock
qa/pipeline.py        configs, prompts, and the ask() pipeline
qa/scoring.py         decides if an answer is correct
evaluate.py           runs the evaluation and writes the report
chat.py / app.py      terminal and web interfaces
tests/                pytest unit tests (run in GitHub Actions on every push)
```

## Things to try next

- Change one setting in `CONFIGS` (`top_k`, `chunk_size`, the prompt), re-run the evaluation, and record whether accuracy went up or down.
- Add harder questions to `eval/questions.jsonl`, such as ones that need facts from two documents.
- Replace BM25 with embedding-based search and compare retrieval hit rates.
- Swap in your own documents (course notes, a product manual) by putting `.md` files in `docs/`.

## Notes

The documents in `docs/` are sample help-center content written for this demo; they don't describe real Kikapu policies.
