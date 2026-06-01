"""
chembl.py
----------
Query ChEMBL for drug/compound information and bioactivity data.

ChEMBL is the definitive resource for drug-like compounds,
bioactivity measurements, and approved drug information.
Completely free REST API.

API docs: https://www.ebi.ac.uk/chembl/api/data/
"""

import time
import requests
from loguru import logger

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
HEADERS = {"Accept": "application/json"}


def lookup_drug(
    drug_name: str,
    max_results: int = 5,
) -> dict:
    """
    Look up a drug or compound in ChEMBL by name.

    Parameters
    ----------
    drug_name : str
        Drug name or compound name (e.g., "osimertinib", "imatinib", "pembrolizumab").
    max_results : int
        Max results to return.

    Returns
    -------
    dict with keys: query, drugs (list of dicts)
    """
    logger.debug(f"[chembl] Looking up drug: {drug_name!r}")

    params = {
        "pref_name__icontains": drug_name,
        "format": "json",
        "limit": max_results,
    }

    try:
        # Search molecule endpoint
        response = requests.get(
            f"{BASE_URL}/molecule",
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        molecules = data.get("molecules", [])

        if not molecules:
            # Try broader search
            params = {
                "molecule_synonyms__synonym__icontains": drug_name,
                "format": "json",
                "limit": max_results,
            }
            response = requests.get(
                f"{BASE_URL}/molecule",
                params=params,
                headers=HEADERS,
                timeout=15,
            )
            data = response.json()
            molecules = data.get("molecules", [])

        drugs = []
        for mol in molecules:
            chembl_id = mol.get("molecule_chembl_id", "N/A")
            pref_name = mol.get("pref_name", "N/A")
            mol_type = mol.get("molecule_type", "N/A")
            max_phase = mol.get("max_phase", 0)

            # Properties
            props = mol.get("molecule_properties", {}) or {}
            mol_weight = props.get("full_mwt", "N/A")
            alogp = props.get("alogp", "N/A")

            # Structures
            structures = mol.get("molecule_structures", {}) or {}
            smiles = structures.get("canonical_smiles", "N/A")
            if smiles and len(smiles) > 100:
                smiles = smiles[:100] + "..."

            # Synonyms
            synonyms_raw = mol.get("molecule_synonyms", []) or []
            synonyms = [s.get("molecule_synonym", "") for s in synonyms_raw[:5]]

            # Indication
            indication_class = mol.get("indication_class", "")

            # Cross references
            cross_refs = mol.get("cross_references", []) or []
            drugbank_id = next(
                (r.get("xref_id") for r in cross_refs if r.get("xref_src") == "DrugBank"),
                None,
            )

            drug_entry = {
                "chembl_id": chembl_id,
                "name": pref_name,
                "molecule_type": mol_type,
                "max_clinical_phase": max_phase,
                "is_approved": max_phase == 4,
                "indication_class": indication_class,
                "synonyms": [s for s in synonyms if s],
                "molecular_weight": mol_weight,
                "alogp": alogp,
                "smiles": smiles,
                "drugbank_id": drugbank_id,
                "url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/",
            }

            # Fetch mechanism of action (separate endpoint)
            moa_data = _get_mechanism(chembl_id)
            drug_entry["mechanisms_of_action"] = moa_data

            drugs.append(drug_entry)

        time.sleep(0.5)
        logger.info(f"[chembl] Found {len(drugs)} results for {drug_name!r}")

        return {
            "query": drug_name,
            "drugs": drugs,
        }

    except Exception as e:
        logger.error(f"[chembl] Error: {e}")
        return {
            "query": drug_name,
            "drugs": [],
            "error": str(e),
        }


def _get_mechanism(chembl_id: str) -> list:
    """Fetch mechanism of action for a ChEMBL compound."""
    try:
        response = requests.get(
            f"{BASE_URL}/mechanism",
            params={"molecule_chembl_id": chembl_id, "format": "json", "limit": 5},
            headers=HEADERS,
            timeout=10,
        )
        if response.status_code != 200:
            return []
        data = response.json()
        mechanisms = data.get("mechanisms", [])
        return [
            {
                "mechanism": m.get("mechanism_of_action", ""),
                "target_name": m.get("target_name", ""),
                "action_type": m.get("action_type", ""),
            }
            for m in mechanisms
        ]
    except Exception:
        return []


# JSON schema for Groq tool calling
SCHEMA = {
    "type": "function",
    "function": {
        "name": "lookup_drug",
        "description": (
            "Look up a drug or compound in ChEMBL. Returns the drug's "
            "mechanism of action, molecular type (small molecule vs. "
            "antibody), clinical phase, approval status, synonyms, "
            "and chemical properties. Use this to understand what a "
            "specific drug does and how it works. Good for: "
            "'what is the mechanism of osimertinib?', "
            "'is imatinib approved?', 'what type of molecule is pembrolizumab?'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": (
                        "Drug or compound name. Use INN (International Nonproprietary "
                        "Name) when possible. Examples: 'osimertinib', 'imatinib', "
                        "'pembrolizumab', 'sotorasib', 'trastuzumab'"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max drugs to return (default 3).",
                    "default": 3,
                },
            },
            "required": ["drug_name"],
        },
    },
}
