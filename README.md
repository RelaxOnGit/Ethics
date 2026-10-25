# SourceLens (Gemini)

## Run on Windows

1. Install Node.js (Node 20+ recommended).
2. Copy `.env.example` to a new file named `.env`.
3. Put your Gemini API key in `.env`:
   `GEMINI_API_KEY=your_key_here`
4. Open CMD in this folder and run:
   ```bat
   npm install
   npm start
   ```
5. Open http://localhost:3000

## AI model

The backend currently uses `gemini-3.8-flash`. The frontend reads the configured model name from `/api/info`, so the label is kept in sync with `server.js`.

## Speed features

- **Streaming**: text appears while Gemini is still writing (`/api/analyze-stream`, Server-Sent Events). `/api/analyze` remains as a non-streaming fallback.
- **DOI shortcut**: DOIs are looked up in Crossref first (fast, real metadata). When found, Gemini runs without web tools.
- **Cache**: repeated sources are answered instantly (in memory, 24 hours, cleared on restart).
- **Short answers**: the prompt asks for about 300 words and `MAX_OUTPUT_TOKENS` caps the length.
- **Timeout**: each analysis stops after 45 seconds, and stops early if you close the tab.

Optional settings in `.env`:

```
GEMINI_MODEL=            # e.g. a flash-lite model for more speed
ANALYZE_TIMEOUT_MS=45000
MAX_OUTPUT_TOKENS=2000
THINKING_BUDGET=         # only if your model supports it; lower = faster
```

## Appearance

Use the button in the top-right corner to switch between Light Mode and Dark Mode. The chosen theme is saved in the browser for future visits.

## Current scope

The page analyzes a public URL, DOI, or paper title. PDF upload is not connected yet.

## Security

- Never share `.env` or put the API key in `index.html`.
- `.env` and `node_modules` are excluded from the ZIP.
- If a real key was included in a shared ZIP, revoke it and create a replacement.
