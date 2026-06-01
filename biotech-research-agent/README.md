# 🧬 Biotech Research Agent

> **Ask complex biotech research questions. Get sourced, structured answers backed by real scientific databases.**

"What phase 2/3 trials exist for KRAS inhibitors in pancreatic cancer, and what is the mechanism of action?" — a question that requires searching literature, clinical databases, and drug-target data. A single LLM call guesses. This agent **finds out**.

---

## How It Works

This is a **ReAct agent** (Reason + Act): the LLM loops between reasoning about what it needs and calling tools to get it, until it has enough evidence to answer.

```
User question
     │
     ▼
┌─────────────────────────────────────────────┐
│              ReAct Agent Loop               │
│                                             │
│  [Groq LLM] ← query + tool definitions     │
│       │                                     │
│       ├─ search_pubmed ──────► PubMed       │
│       ├─ search_trials ──────► ClinTrials   │
│       ├─ lookup_protein ─────► UniProt      │
│       ├─ query_drug_targets ─► OpenTargets  │
│       ├─ lookup_drug ────────► ChEMBL       │
│       └─ search_genes ───────► NCBI Gene    │
│                                             │
│  [Groq LLM] synthesizes final answer        │
└─────────────────────────────────────────────┘
     │
     ▼
Sourced answer + tool trace + optional report
```

---

## Tools (all free, no paid APIs)

| Tool | Database | What it answers |
|---|---|---|
| `search_pubmed` | PubMed / NCBI Entrez | Literature: papers, abstracts, evidence |
| `search_trials` | ClinicalTrials.gov v2 | Active/completed trials, phase, status, sponsors |
| `lookup_protein` | UniProt REST | Protein function, disease links, pathways |
| `query_drug_targets` | OpenTargets GraphQL | Drug→target→disease associations, clinical evidence |
| `lookup_drug` | ChEMBL REST | Compound info, mechanism of action, approval status |
| `search_genes` | NCBI Gene / Entrez | Gene function, chromosomal location, related diseases |

---

## LLM: Groq

**Groq** runs open LLMs at ~500 tokens/second — fast enough that multi-tool agent loops feel instant. Free tier is sufficient for portfolio use.

Default model: `llama3-70b-8192` (best tool calling + reasoning)

Get a free API key at [console.groq.com](https://console.groq.com)

---

## Example Queries

- *"What are the approved EGFR inhibitors for non-small cell lung cancer and what are their resistance mechanisms?"*
- *"Find phase 3 trials for checkpoint inhibitors in triple-negative breast cancer"*
- *"What is the role of the BRCA2 gene in DNA repair and which drugs target this pathway?"*
- *"What does the clinical evidence say about CAR-T therapy for B-cell lymphoma?"*
- *"Compare the mechanisms of osimertinib vs erlotinib"*

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set up credentials
cp .env.example .env
# Edit .env: add GROQ_API_KEY and NCBI_EMAIL

# 3. Launch
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
biotech-research-agent/
├── README.md
├── config.yaml              ← model, tools, rate limits
├── requirements.txt
├── .env.example             ← GROQ_API_KEY, NCBI_EMAIL
├── Makefile
│
├── src/
│   ├── agent.py             ← ReAct loop + Groq tool calling
│   ├── utils.py             ← caching, retry, rate limiting
│   ├── report_generator.py  ← format answer → structured report
│   └── tools/
│       ├── __init__.py      ← tool registry + JSON schemas
│       ├── pubmed.py
│       ├── clinical_trials.py
│       ├── uniprot.py
│       ├── opentargets.py
│       ├── chembl.py
│       └── ncbi_gene.py
│
├── app/
│   └── streamlit_app.py     ← chat UI + tool trace panel
│
└── notebooks/
    └── 01_Tool_Testing.ipynb  ← test each tool individually
```

---

## References

- Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.
- Groq documentation: [console.groq.com/docs](https://console.groq.com/docs)
- ClinicalTrials.gov API v2: [clinicaltrials.gov/data-api](https://clinicaltrials.gov/data-api/v2)
- OpenTargets Platform: [platform.opentargets.org](https://platform.opentargets.org)
