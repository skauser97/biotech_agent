"""
ClinicalTrials.gov v2 API tool — drug pipelines and trial status.
Free, no API key required.
"""

import requests
from langchain_core.tools import tool


BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


@tool
def search_clinical_trials(query: str, max_results: int = 5, status: str = "all") -> str:
    """
    Search ClinicalTrials.gov for drug trials and clinical studies.
    Use for: drug pipeline status, Phase I/II/III trials, FDA approval pipeline,
    specific drug names, diseases, or company-sponsored trials.

    Args:
        query: Drug name, disease, or company (e.g. 'CRISPR sickle cell', 'Ozempic obesity', 'Recursion')
        max_results: Number of trials to return (default 5, max 10)
        status: Filter by status — 'recruiting', 'completed', 'active', or 'all' (default)
    """
    max_results = min(int(max_results), 10)  # coerce str→int in case model passes "5"

    params = {
        "query.term": query,
        "pageSize": max_results,
        "format": "json",
        "fields": "NCTId,BriefTitle,OverallStatus,Phase,StartDate,CompletionDate,LeadSponsorName,BriefSummary,Condition,InterventionName",
    }

    # Add status filter
    status_map = {
        "recruiting": "RECRUITING",
        "completed": "COMPLETED",
        "active": "ACTIVE_NOT_RECRUITING",
    }
    if status.lower() in status_map:
        params["filter.overallStatus"] = status_map[status.lower()]

    r = requests.get(BASE_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    studies = data.get("studies", [])
    total = data.get("totalCount", 0)

    if not studies:
        return f"No clinical trials found for: '{query}'"

    output = [f"**ClinicalTrials.gov: '{query}'** — {total} total trials, showing {len(studies)}\n"]
    for i, study in enumerate(studies, 1):
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        sponsor = proto.get("sponsorCollaboratorsModule", {})
        desc = proto.get("descriptionModule", {})
        conditions = proto.get("conditionsModule", {})
        interventions = proto.get("armsInterventionsModule", {})

        nct_id = ident.get("nctId", "?")
        title = ident.get("briefTitle", "No title")
        trial_status = status_mod.get("overallStatus", "?")
        phases = design.get("phases", [])
        phase_str = ", ".join(phases) if phases else "N/A"
        start = status_mod.get("startDateStruct", {}).get("date", "?")
        completion = status_mod.get("completionDateStruct", {}).get("date", "?")
        lead_sponsor = sponsor.get("leadSponsor", {}).get("name", "?")
        summary = desc.get("briefSummary", "")[:400]
        if len(summary) > 400:
            summary += "..."
        conds = ", ".join(conditions.get("conditions", [])[:3])

        output.append(
            f"{i}. **{title}**\n"
            f"   NCT: {nct_id} | Phase: {phase_str} | Status: {trial_status}\n"
            f"   Sponsor: {lead_sponsor} | Start: {start} | Est. completion: {completion}\n"
            f"   Conditions: {conds}\n"
            f"   {summary}\n"
            f"   🔗 https://clinicaltrials.gov/study/{nct_id}\n"
        )
    return "\n".join(output)