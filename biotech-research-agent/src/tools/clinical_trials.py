"""
clinical_trials.py
-------------------
Search ClinicalTrials.gov via the v2 REST API.

Completely free, no authentication required.
API docs: https://clinicaltrials.gov/data-api/v2

Returns trial metadata: NCT ID, title, phase, status,
sponsor, conditions, interventions, enrollment, dates.
"""

import time
import requests
from typing import Optional, List
from loguru import logger

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

HEADERS = {
    "User-Agent": (
        "BiotechResearchAgent/1.0 (portfolio project; "
        "contact: research@example.com)"
    )
}

STATUS_MAP = {
    "RECRUITING": "Recruiting",
    "ACTIVE_NOT_RECRUITING": "Active (not recruiting)",
    "COMPLETED": "Completed",
    "TERMINATED": "Terminated",
    "WITHDRAWN": "Withdrawn",
    "SUSPENDED": "Suspended",
    "NOT_YET_RECRUITING": "Not yet recruiting",
    "ENROLLING_BY_INVITATION": "Enrolling by invitation",
}


def search_trials(
    query: str,
    max_results: int = 10,
    status_filter: Optional[List[str]] = None,
    phase_filter: Optional[List[str]] = None,
) -> dict:
    """
    Search ClinicalTrials.gov for studies matching a query.

    Parameters
    ----------
    query : str
        Free-text search (condition, drug, intervention, sponsor).
    max_results : int
        Max trials to return (default 10).
    status_filter : list of str, optional
        Filter by status. Options: "RECRUITING", "COMPLETED",
        "TERMINATED", "ACTIVE_NOT_RECRUITING", etc.
        Pass None for all statuses.
    phase_filter : list of str, optional
        Filter by phase: "PHASE1", "PHASE2", "PHASE3", "PHASE4".

    Returns
    -------
    dict with keys: query, total_found, trials (list of dicts)
    """
    logger.debug(f"[trials] Searching: {query!r}")

    params = {
        "query.term": query,
        "pageSize": min(max_results, 100),
        "format": "json",
        "fields": (
            "NCTId,BriefTitle,OverallStatus,Phase,StudyType,"
            "LeadSponsorName,Condition,InterventionName,"
            "EnrollmentCount,StartDate,CompletionDate,"
            "BriefSummary,EligibilityCriteria"
        ),
    }

    if status_filter:
        params["filter.overallStatus"] = "|".join(status_filter)

    try:
        response = requests.get(
            BASE_URL, params=params, headers=HEADERS, timeout=20
        )
        response.raise_for_status()
        data = response.json()

        studies = data.get("studies", [])
        total = data.get("totalCount", len(studies))

        trials = []
        for study in studies:
            proto = study.get("protocolSection", {})
            id_module = proto.get("identificationModule", {})
            status_module = proto.get("statusModule", {})
            design_module = proto.get("designModule", {})
            sponsor_module = proto.get("sponsorCollaboratorsModule", {})
            conditions_module = proto.get("conditionsModule", {})
            interventions_module = proto.get("armsInterventionsModule", {})
            desc_module = proto.get("descriptionModule", {})
            eligibility_module = proto.get("eligibilityModule", {})

            nct_id = id_module.get("nctId", "N/A")
            title = id_module.get("briefTitle", "No title")
            status = STATUS_MAP.get(
                status_module.get("overallStatus", ""),
                status_module.get("overallStatus", "Unknown")
            )
            phase = design_module.get("phases", ["N/A"])
            if isinstance(phase, list):
                phase = ", ".join(phase)

            sponsor = sponsor_module.get("leadSponsor", {}).get("name", "Unknown")
            conditions = conditions_module.get("conditions", [])
            interventions = [
                i.get("name", "")
                for i in interventions_module.get("interventions", [])
            ]

            enrollment = design_module.get("enrollmentInfo", {}).get("count", "N/A")
            start_date = status_module.get("startDateStruct", {}).get("date", "N/A")
            completion_date = status_module.get(
                "primaryCompletionDateStruct", {}
            ).get("date", "N/A")

            summary = desc_module.get("briefSummary", "")[:500]

            trials.append({
                "nct_id": nct_id,
                "title": title,
                "status": status,
                "phase": phase,
                "sponsor": sponsor,
                "conditions": conditions[:3],
                "interventions": interventions[:5],
                "enrollment": enrollment,
                "start_date": start_date,
                "completion_date": completion_date,
                "summary": summary,
                "url": f"https://clinicaltrials.gov/study/{nct_id}",
            })

        # Optional phase filter (post-process since API filter is less reliable)
        if phase_filter:
            phase_set = set(p.upper() for p in phase_filter)
            trials = [
                t for t in trials
                if any(p.upper() in t["phase"].upper() for p in phase_set)
            ]

        logger.info(f"[trials] Found {total} total, returning {len(trials)}")
        time.sleep(0.5)

        return {
            "query": query,
            "total_found": total,
            "trials": trials,
        }

    except Exception as e:
        logger.error(f"[trials] Error: {e}")
        return {
            "query": query,
            "total_found": 0,
            "trials": [],
            "error": str(e),
        }


# JSON schema for Groq tool calling
SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_trials",
        "description": (
            "Search ClinicalTrials.gov for clinical studies. Use this to find "
            "what trials exist for a drug, disease, or target. Returns trial phase, "
            "status (recruiting/completed/terminated), sponsor, enrollment size, "
            "and trial dates. Good for: 'what trials exist for X?', "
            "'is Y in phase 3?', 'which companies are testing Z?'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search term: drug name, disease, condition, or intervention. "
                        "Examples: 'osimertinib lung cancer', 'CAR-T lymphoma', "
                        "'KRAS G12C pancreatic'"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max trials to return (default 10).",
                    "default": 10,
                },
                "status_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Filter by trial status. Options: RECRUITING, COMPLETED, "
                        "ACTIVE_NOT_RECRUITING, TERMINATED. Leave empty for all."
                    ),
                },
                "phase_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Filter by phase: PHASE1, PHASE2, PHASE3, PHASE4. "
                        "Leave empty for all phases."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}
