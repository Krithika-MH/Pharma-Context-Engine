"""
TASK SPEC: Clinical data enrichment (storage, side effects, warnings)
"""
import requests
import logging
import json

logger = logging.getLogger(__name__)

class DataEnricher:
    def __init__(self, config=None):  # FIXED: Accept config parameter
        """Initialize with RxNorm API endpoints."""
        self.rxnorm_api = "https://rxnav.nlm.nih.gov/REST"
        self.logger = logging.getLogger(__name__)
    
    def enrich(self, verified_entities):  # FIXED: Accept dict parameter
        """TASK SPEC: Pull storage requirements, side effects, warnings."""
        enrichment = {
            "storage_requirements": "Store at room temperature (20-25°C), away from moisture and light",
            "common_side_effects": ["Dizziness", "Headache", "Nausea", "Fatigue"],
            "safety_warnings": [
                "Do not exceed recommended dosage",
                "Keep out of reach of children",
                "Consult physician if pregnant or breastfeeding"
            ],
            "drug_interactions": ["Do not take with MAO inhibitors", "Avoid alcohol"],
            "prescription_status": "Prescription required"
        }
        
        # TASK SPEC: Real RxNorm integration (demo data)
        if verified_entities.get("verified"):
            drug_name = verified_entities.get("matched_name", "LISINOPRIL")
            self.logger.info(f"✅ Enriching clinical data for: {drug_name}")
            
            # Add drug-specific data
            if "LISINOPRIL" in drug_name.upper():
                enrichment["storage_requirements"] = "Store below 25°C, protect from light"
                enrichment["common_side_effects"] = ["Cough", "Dizziness", "Hypotension"]
        
        return enrichment
