"""
AirRoute AI - Multi-Issue Detection & Intelligent Department Routing Engine
Segments passenger reviews into constituent clauses, extracts multiple operational issues,
determines sentiment & priority, and routes complaints to responsible airline departments.
"""

import re
import logging
from typing import List, Dict, Any, Optional
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Department Lexicon & Pattern Mapping
DEPARTMENT_KEYWORDS = {
    "Flight Operations": {
        "keywords": [
            "delay", "delayed", "late", "cancelled", "cancel", "cancelling", "stranded",
            "divert", "diverted", "tarmac", "hours late", "missed connection", "mechanical",
            "pilot", "takeoff", "landing", "weather", "snowstorm", "hold"
        ],
        "priority_triggers": ["stranded", "cancelled", "hours late", "missed connection", "mechanical"]
    },
    "Baggage Services": {
        "keywords": [
            "luggage", "bag", "bags", "baggage", "suitcase", "lost bag", "damaged bag",
            "broken bag", "carousel", "missing bag", "lost luggage", "damaged", "broken"
        ],
        "priority_triggers": ["lost", "damaged", "broken", "missing bag", "lost luggage"]
    },
    "Reservations & Ticketing": {
        "keywords": [
            "book", "booking", "rebook", "rebooked", "refund", "charge", "charged",
            "credit", "ticket", "reservation", "change flight", "confirmation", "price",
            "fare", "receipt", "overbooked", "cancel reservation"
        ],
        "priority_triggers": ["refund", "double charge", "charged twice", "overbooked"]
    },
    "Customer Experience": {
        "keywords": [
            "flight attendant", "crew", "cabin crew", "agent", "staff", "csr", "representative",
            "customer service", "rude", "unprofessional", "polite", "helpful", "attitude",
            "unfriendly", "gate agent", "manager", "assistance", "desk"
        ],
        "priority_triggers": ["rude", "harassed", "unprofessional", "ignored"]
    },
    "In-flight Services": {
        "keywords": [
            "food", "meal", "snack", "drink", "beverage", "seat", "legroom", "wifi",
            "entertainment", "tv", "screen", "audio", "movie", "cabin", "temperature",
            "dirty", "headphone", "plug", "outlet", "power"
        ],
        "priority_triggers": ["dirty", "broken seat", "no water"]
    },
    "Digital Support": {
        "keywords": [
            "website", "site", "app", "mobile app", "online", "login", "server",
            "error message", "bug", "crash", "kiosk", "check-in online", "boarding pass"
        ],
        "priority_triggers": ["crash", "system down", "error message", "can't check in"]
    }
}

SENTIMENT_INTENSITY_NEGATIVE = [
    "terrible", "horrible", "worst", "awful", "unacceptable", "disaster", "disgraceful",
    "disgusted", "appalling", "furious", "pathetic", "nightmare", "sucks", "hate"
]


class AirlineReviewRouter:
    """
    Intelligent routing system capable of parsing multi-issue compound reviews.
    """

    def __init__(
        self,
        model_path: str = "models/baseline_logistic_regression.joblib",
        vectorizer_path: str = "models/tfidf_vectorizer.joblib"
    ):
        self.model = None
        self.vectorizer = None
        self.sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
        self._load_models(model_path, vectorizer_path)

    def _load_models(self, model_path: str, vectorizer_path: str):
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            logger.info("Successfully loaded ML models for review router.")
        except Exception as e:
            logger.warning(f"Could not load ML models from disk ({e}). Falling back to rule-based routing.")

    def predict_overall_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Predicts overall review sentiment and class probabilities using ML model.
        """
        if self.model is not None and self.vectorizer is not None:
            features = self.vectorizer.transform([text])
            pred_idx = int(self.model.predict(features)[0])
            probs = self.model.predict_proba(features)[0]
            
            return {
                "sentiment": self.sentiment_labels.get(pred_idx, "neutral"),
                "confidence": round(float(probs[pred_idx]), 4),
                "probabilities": {
                    "negative": round(float(probs[0]), 4),
                    "neutral": round(float(probs[1]), 4),
                    "positive": round(float(probs[2]), 4)
                }
            }
        
        # Rule-based fallback
        lower = text.lower()
        if any(w in lower for w in ["worst", "terrible", "bad", "delay", "lost", "broken", "rude", "cancel"]):
            return {"sentiment": "negative", "confidence": 0.85, "probabilities": {"negative": 0.85, "neutral": 0.1, "positive": 0.05}}
        elif any(w in lower for w in ["great", "awesome", "good", "thank", "helpful", "love", "best"]):
            return {"sentiment": "positive", "confidence": 0.85, "probabilities": {"negative": 0.05, "neutral": 0.1, "positive": 0.85}}
        return {"sentiment": "neutral", "confidence": 0.70, "probabilities": {"negative": 0.2, "neutral": 0.7, "positive": 0.1}}

    def split_into_clauses(self, text: str) -> List[str]:
        """
        Splits compound sentences into meaningful sub-clauses using punctuation & discourse markers.
        Example: 'Flight was delayed, baggage was damaged, but crew was great'
        -> ['Flight was delayed', 'baggage was damaged', 'crew was great']
        """
        # Replace conjunction connectors with delimiter
        delimiters = r"(?:\s*;\s*|\s*,\s*(?:but|however|although|while|and|yet)\s*|\s+(?:but|however|although|while)\s*|\. |\n+)"
        clauses = re.split(delimiters, text, flags=re.IGNORECASE)
        
        # Clean clauses
        cleaned_clauses = []
        for c in clauses:
            cleaned = re.sub(r"^[@\s,.-]+", "", c).strip()
            if len(cleaned) > 5:
                cleaned_clauses.append(cleaned)
        
        return cleaned_clauses if cleaned_clauses else [text]

    def _determine_clause_sentiment(self, clause: str, overall_sentiment: str) -> str:
        """
        Determines sentiment polarity for a specific clause.
        """
        c_lower = clause.lower()
        pos_words = ["helpful", "great", "awesome", "polite", "friendly", "good", "loved", "fast", "thanks", "impressed"]
        neg_words = ["delayed", "delay", "lost", "damaged", "broken", "rude", "cancelled", "cancel", "terrible", "worst", "unacceptable", "error", "nightmare"]
        
        pos_count = sum(1 for w in pos_words if w in c_lower)
        neg_count = sum(1 for w in neg_words if w in c_lower)

        if neg_count > pos_count:
            return "negative"
        if pos_count > neg_count:
            return "positive"
        return overall_sentiment

    def _determine_priority(self, department: str, clause: str, sentiment: str) -> str:
        """
        Determines ticket priority level (URGENT, HIGH, MEDIUM, LOW).
        """
        if sentiment == "positive" or sentiment == "neutral":
            return "LOW"

        c_lower = clause.lower()

        # Check urgent triggers
        if any(w in c_lower for w in ["stranded", "hotel", "missed connection", "emergency", "furious", "police"]):
            return "URGENT"

        dept_info = DEPARTMENT_KEYWORDS.get(department, {})
        priority_triggers = dept_info.get("priority_triggers", [])
        
        # Check high triggers
        if any(trig in c_lower for trig in priority_triggers):
            return "HIGH"

        # Check intensity words
        if any(w in c_lower for w in SENTIMENT_INTENSITY_NEGATIVE):
            return "HIGH"

        return "MEDIUM"

    def analyze_and_route(self, review_text: str, airline_name: Optional[str] = "Unknown") -> Dict[str, Any]:
        """
        Main routing pipeline:
        1. Predict overall sentiment
        2. Break into sub-clauses
        3. Match departments
        4. Calculate priority per issue
        5. Return structured tickets & routing metadata
        """
        overall = self.predict_overall_sentiment(review_text)
        clauses = self.split_into_clauses(review_text)
        
        extracted_issues = []
        matched_depts = set()

        for clause in clauses:
            c_lower = clause.lower()
            matched_for_clause = False

            for dept, data in DEPARTMENT_KEYWORDS.items():
                keywords = data["keywords"]
                # Match keywords in clause
                matched_kw = [kw for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", c_lower)]
                
                if matched_kw:
                    clause_sentiment = self._determine_clause_sentiment(clause, overall["sentiment"])
                    priority = self._determine_priority(dept, clause, clause_sentiment)
                    is_actionable = clause_sentiment == "negative"

                    issue_entry = {
                        "issue_description": clause,
                        "department": dept,
                        "matched_keywords": matched_kw,
                        "sentiment": clause_sentiment,
                        "priority": priority,
                        "is_actionable": is_actionable
                    }
                    extracted_issues.append(issue_entry)
                    matched_depts.add(dept)
                    matched_for_clause = True

            # If no keyword matched, but clause is present and negative, route to Customer Experience
            if not matched_for_clause and overall["sentiment"] == "negative":
                extracted_issues.append({
                    "issue_description": clause,
                    "department": "Customer Experience",
                    "matched_keywords": ["general feedback"],
                    "sentiment": "negative",
                    "priority": "MEDIUM",
                    "is_actionable": True
                })
                matched_depts.add("Customer Experience")

        # Fallback if no issues extracted at all
        if not extracted_issues:
            dept = "Customer Experience" if overall["sentiment"] != "positive" else "Customer Experience"
            extracted_issues.append({
                "issue_description": review_text,
                "department": dept,
                "matched_keywords": ["general"],
                "sentiment": overall["sentiment"],
                "priority": "LOW" if overall["sentiment"] == "positive" else "MEDIUM",
                "is_actionable": overall["sentiment"] == "negative"
            })
            matched_depts.add(dept)

        return {
            "review_text": review_text,
            "airline": airline_name,
            "overall_sentiment": overall["sentiment"],
            "confidence": overall["confidence"],
            "probabilities": overall["probabilities"],
            "total_issues_detected": len(extracted_issues),
            "target_departments": list(matched_depts),
            "extracted_issues": extracted_issues
        }


# Global singleton instance
router = AirlineReviewRouter()


if __name__ == "__main__":
    sample_review = (
        "@united My flight was delayed by four hours and you lost my baggage in Chicago, "
        "but the cabin crew was very polite and helpful!"
    )
    res = router.analyze_and_route(sample_review, airline_name="United")
    import pprint
    print("\n--- MULTI-ISSUE ROUTING TEST RESULT ---")
    pprint.pprint(res)
