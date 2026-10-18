# Invoice extraction FastAPI service

## Render configuration
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Root directory: blank when these files are at the repository root

## Endpoints
- `GET /health`
- `POST /extract` with JSON body `{"text":"..."}`

The parser handles labeled invoice fields and common vendor company names. Optional Ollama support is enabled by setting:
- `OLLAMA_BASE_URL` to an Ollama server reachable from the deployed service
- `OLLAMA_MODEL` to the model name available on that server

A local Ollama instance running on your own computer is not reachable from Render unless you deliberately expose it over a secure network/tunnel. Without Ollama configured, the endpoint uses text parsing and returns 422 if required fields cannot be extracted.
