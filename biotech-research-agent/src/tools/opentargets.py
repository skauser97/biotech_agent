"""
opentargets.py
---------------
Query the Open Targets Platform for drug-target-disease associations.

Open Targets integrates genetic, genomic, and chemical data to provide
evidence-based drug-target associations. Free GraphQL API.

Platform: https://platform.opentargets.org
API docs: https://platform-docs.opentargets.org/data-access/graphql-api
"""

import time
import requests
from loguru import logger

GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
HEADERS = {"Content-Type": "application/json"}


def _run_query(query: str, variables: dict) -> dict:
    """Execute a GraphQL query against Open Targets."""
    payload = {"query": query, "variables": variables}
    response = requests.post(
        GRAPHQL_URL, json=payload, headers=HEADERS, timeout=20
    )
    response.raise_for_status()
    return response.json()


def query_drug_targets(
    target_gene: str,
    max_drugs: int = 10,
    min_phase: int = 0,
) -> dict:
    """
    Find drugs associated with a gene target and their clinical status.

    Parameters
    ----------
    target_gene : str
        HGNC gene symbol (e.g., "EGFR", "KRAS", "BRCA1", "PDCD1").
    max_drugs : int
        Maximum drugs to return.
    min_phase : int
        Minimum clinical phase (0 = include preclinical, 3 = Phase 3+ only).

    Returns
    -------
    dict with keys: target_gene, target_info, drugs (list of dicts)
    """
    logger.debug(f"[opentargets] Querying drugs for target: {target_gene!r}")

    # Step 1: Get target ID from gene symbol
    target_search_query = """
    query TargetSearch($q: String!) {
        search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 1}) {
            hits {
                id
                name
                entity
                object {
                    ... on Target {
                        id
                        approvedSymbol
                        approvedName
                        biotype
                        functionDescriptions
                    }
                }
            }
        }
    }
    """
    try:
        search_result = _run_query(target_search_query, {"q": target_gene})
        hits = search_result.get("data", {}).get("search", {}).get("hits", [])

        if not hits:
            return {
                "target_gene": target_gene,
                "target_info": {},
                "drugs": [],
                "message": f"Target '{target_gene}' not found in Open Targets.",
            }

        target_obj = hits[0].get("object", {})
        target_id = target_obj.get("id", "")
        target_name = target_obj.get("approvedName", target_gene)
        functions = target_obj.get("functionDescriptions", [])

        # Step 2: Get drugs for this target
        drugs_query = """
        query TargetDrugs($targetId: String!, $size: Int!) {
            target(ensemblId: $targetId) {
                id
                approvedSymbol
                knownDrugs(size: $size) {
                    count
                    rows {
                        drug {
                            id
                            name
                            maximumClinicalTrialPhase
                            isApproved
                            mechanismsOfAction {
                                rows {
                                    mechanismOfAction
                                    actionType
                                }
                            }
                            indications {
                                rows {
                                    disease {
                                        name
                                    }
                                    maxPhaseForIndication
                                }
                            }
                        }
                        phase
                        status
                        disease {
                            name
                        }
                    }
                }
            }
        }
        """

        drugs_result = _run_query(drugs_query, {"targetId": target_id, "size": max_drugs * 2})
        target_data = drugs_result.get("data", {}).get("target", {})
        known_drugs = target_data.get("knownDrugs", {})
        drug_rows = known_drugs.get("rows", [])
        total_drugs = known_drugs.get("count", 0)

        drugs = []
        seen_drug_ids = set()

        for row in drug_rows:
            drug = row.get("drug", {})
            drug_id = drug.get("id", "")
            if drug_id in seen_drug_ids:
                continue
            seen_drug_ids.add(drug_id)

            phase = row.get("phase", 0) or 0
            if phase < min_phase:
                continue

            drug_name = drug.get("name", "Unknown")
            is_approved = drug.get("isApproved", False)
            max_phase = drug.get("maximumClinicalTrialPhase", phase)

            mechanisms = drug.get("mechanismsOfAction", {}).get("rows", [])
            moa_list = [
                f"{m.get('actionType', '')} ({m.get('mechanismOfAction', '')})"
                for m in mechanisms[:2]
            ]

            indications = drug.get("indications", {}).get("rows", [])
            indication_names = [
                f"{i.get('disease', {}).get('name', '')} (Phase {i.get('maxPhaseForIndication', '?')})"
                for i in indications[:4]
                if i.get("disease", {}).get("name")
            ]

            disease_name = row.get("disease", {}).get("name", "")

            drugs.append({
                "drug_id": drug_id,
                "name": drug_name,
                "is_approved": is_approved,
                "max_clinical_phase": max_phase,
                "phase_for_this_target": phase,
                "status": row.get("status", "N/A"),
                "disease": disease_name,
                "mechanisms_of_action": moa_list,
                "other_indications": indication_names,
                "url": f"https://platform.opentargets.org/drug/{drug_id}",
            })

            if len(drugs) >= max_drugs:
                break

        time.sleep(0.5)
        logger.info(
            f"[opentargets] {target_gene}: {total_drugs} known drugs, "
            f"returning {len(drugs)}"
        )

        return {
            "target_gene": target_gene,
            "target_id": target_id,
            "target_name": target_name,
            "target_functions": functions[:2],
            "total_known_drugs": total_drugs,
            "drugs": drugs,
        }

    except Exception as e:
        logger.error(f"[opentargets] Error: {e}")
        return {
            "target_gene": target_gene,
            "drugs": [],
            "error": str(e),
        }


# JSON schema for Groq tool calling
SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_drug_targets",
        "description": (
            "Query Open Targets for drugs associated with a gene target. "
            "Returns drug names, clinical phase, approval status, mechanism "
            "of action, and disease indications. Use this to find what drugs "
            "exist for a given gene/protein target. Good for: 'what drugs "
            "target EGFR?', 'is there anything approved for KRAS?', "
            "'what is the clinical stage of drugs against PD-1?'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_gene": {
                    "type": "string",
                    "description": (
                        "HGNC gene symbol for the target. Examples: "
                        "'EGFR', 'KRAS', 'BRCA1', 'PDCD1' (PD-1), "
                        "'CD274' (PD-L1), 'ERBB2' (HER2), 'ALK', 'MET'"
                    ),
                },
                "max_drugs": {
                    "type": "integer",
                    "description": "Max drugs to return (default 10).",
                    "default": 10,
                },
                "min_phase": {
                    "type": "integer",
                    "description": (
                        "Minimum clinical phase. 0=all, 1=Phase1+, "
                        "2=Phase2+, 3=Phase3+, 4=approved."
                    ),
                    "default": 0,
                },
            },
            "required": ["target_gene"],
        },
    },
}
