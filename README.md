# 🧬 Biotech Research Assistant

Your personal on-demand research assistant for **AI × Biotech** — papers, drugs, clinical trials, company intelligence, autonomous labs, and GPU compute science.

## Quick Start

### 1. Install dependencies
```bash
cd biotech_agent
pip install -r requirements.txt
```

### 2. Configure your LLM
```bash
cp .env.example .env
```

Edit `.env` and choose your backend:

**Option A — Groq (recommended: fast, free tier)**
```
LLM_BACKEND=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

**Option B — Ollama (local, private)**
```bash
ollama serve                    # start Ollama
ollama pull qwen2.5:14b         # best for tool calling (~9GB)
# or: ollama pull qwen2.5:7b   # lighter option
```
```
LLM_BACKEND=ollama
OLLAMA_MODEL=qwen2.5:14b
```

### 3. Run
```bash
python main.py
```

---

## Tools

| Tool | Source | Best for |
|------|--------|----------|
| `search_pubmed` | NCBI PubMed | Peer-reviewed biomedical papers |
| `search_arxiv` | ArXiv | AI × bio preprints, GPU/compute |
| `search_clinical_trials` | ClinicalTrials.gov | Drug trials, pipeline status |
| `search_openalex` | OpenAlex | Broad research landscape |
| `search_web` | DuckDuckGo | Company news, recent events |
| `search_web_news` | DuckDuckGo News | Breaking news, funding rounds |

All tools are **free with no API key** required.

---

## Example Queries

```
What are the latest papers on protein language models for drug discovery?

Give me an overview of Recursion Pharmaceuticals' current pipeline

What CRISPR trials are currently recruiting for sickle cell disease?

Summarize the state of autonomous laboratories in 2025

Compare Isomorphic Labs and Insilico Medicine approaches to AI drug discovery

What GPU infrastructure is being used for molecular dynamics simulations?

Find papers on self-driving labs and closed-loop experimentation
```

---

## File Structure

```
biotech_agent/
├── main.py              ← CLI entry point (run this)
├── agent.py             ← LangGraph ReAct agent
├── tools/
│   ├── pubmed.py        ← PubMed E-utilities
│   ├── arxiv.py         ← ArXiv API
│   ├── clinical_trials.py  ← ClinicalTrials.gov v2
│   ├── openalex.py      ← OpenAlex
│   └── web_search.py    ← DuckDuckGo (web + news)
├── requirements.txt
├── .env.example
└── .env                 ← your config (gitignored)
```

---

## Coming Next
- [ ] Semantic Scholar (citation graphs, impact scores)
- [ ] Local paper cache (SQLite)
- [ ] Save/export research summaries
- [ ] Notion integration
- [ ] Scheduled daily digest
