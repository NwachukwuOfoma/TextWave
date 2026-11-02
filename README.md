# TextWave

A retrieval augmented generation (RAG) question answering service, built as a Johns Hopkins course project (705.603).

## What it does

- Splits documents using fixed length or sentence chunking
- Embeds chunks with sentence-transformers and indexes them with FAISS
- Reranks results with TF-IDF, a cross encoder, or both in sequence
- Generates answers with the Mistral API
- Serves everything through a Flask `/generate` endpoint, packaged with Docker

## Running it

1. Add your own text documents to a `storage/` folder. The course dataset is not included.
2. Set your API key: `export MISTRAL_API_KEY=...`
3. Build and run:

```bash
docker build -t textwave .
docker run -p 5000:5000 -e MISTRAL_API_KEY textwave
```

## Notes

Parts of this project build on starter code provided for the course. Course data, tests, and written reports are not included in this repository.
