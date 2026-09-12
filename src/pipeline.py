from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from src.settings import load_settings


RISK_TERMS = re.compile(
    r"\b(hacked|fraud|fraudulent|stole|lawsuit|sue|legal|password|security|"
    r"unauthorized|threat|police|scam|phishing|card number)\b",
    re.I,
)
INTENT_PATTERNS = [
    ("complaint", re.compile(r"\b(worst|terrible|ridiculous|unacceptable|disappoint|manager|complaint|arrogant|awful)\b", re.I)),
    ("refund_return", re.compile(r"\b(refund|return|money back|reimbursement|reembolso|remboursement|r[üu]ckerstattung|rimborso)\b", re.I)),
    ("payment_issue", re.compile(r"\b(charged|charge|payment|billing|debit|credit card|amazon pay|paid twice|duplicate)\b", re.I)),
    ("account_access", re.compile(r"\b(log ?in|account|password|sign ?in|locked out|verification code)\b", re.I)),
    ("cancellation_change", re.compile(r"\b(cancel|cancellation|change (my |the )?order|modify (my |the )?order)\b", re.I)),
    ("technical_issue", re.compile(r"\b(app|website|site|error|crash|bug|not working|technical)\b", re.I)),
    (
        "delivery_issue",
        re.compile(
            r"\b(delivery|delivered|package|parcel|tracking|shipment|courier|late|arriv(?:e|ed|ing)|"
            r"colis|paquete|entrega|livraison|liefer\w*|paket|pacco)\b",
            re.I,
        ),
    ),
    ("order_status", re.compile(r"\b(order status|where(?:'s| is) my order|order confirmation|order progress)\b", re.I)),
    ("product_question", re.compile(r"\b(available|availability|compatible|kindle|prime membership|do you offer|can i use)\b", re.I)),
]


@dataclass
class Prediction:
    intent: str
    intent_confidence: float | None
    intent_source: str
    reply: str | None
    decision: str
    reason: str
    evidence: list[dict[str, Any]]


class SupportAgent:
    def __init__(
        self,
        conversations: pd.DataFrame,
        min_confidence: float = 0.55,
        min_similarity: float = 0.45,
        top_k: int = 3,
    ):
        self.conversations = conversations.reset_index(drop=True)
        self.min_confidence = min_confidence
        self.min_similarity = min_similarity
        self.top_k = top_k
        self.intent_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.intent_model = LogisticRegression(max_iter=1000, random_state=7)
        intent_matrix = self.intent_vectorizer.fit_transform(self.conversations.customer_message)
        self.intent_model.fit(intent_matrix, self.conversations.intent)
        self.retrieval_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.retrieval_matrix = self.retrieval_vectorizer.fit_transform(self.conversations.customer_message)

    def _classify(self, message: str) -> tuple[str, float | None, str]:
        for intent, pattern in INTENT_PATTERNS:
            if pattern.search(message):
                return intent, None, "rule"
        matrix = self.intent_vectorizer.transform([message])
        probabilities = self.intent_model.predict_proba(matrix)[0]
        index = probabilities.argmax()
        return str(self.intent_model.classes_[index]), float(probabilities[index]), "model"

    def _retrieve(self, message: str) -> list[dict[str, Any]]:
        query = self.retrieval_vectorizer.transform([message])
        scores = cosine_similarity(query, self.retrieval_matrix)[0]
        indices = scores.argsort()[::-1][: self.top_k]
        return [
            {
                "conversation_id": str(self.conversations.iloc[index].conversation_id),
                "similarity": round(float(scores[index]), 3),
                "customer_message": str(self.conversations.iloc[index].customer_message),
                "agent_reply": str(self.conversations.iloc[index].agent_reply),
            }
            for index in indices
        ]

    def predict(self, message: str) -> Prediction:
        intent, confidence, intent_source = self._classify(message)
        evidence = self._retrieve(message)
        best_similarity = evidence[0]["similarity"] if evidence else 0.0
        reasons: list[str] = []
        if confidence is not None and confidence < self.min_confidence and best_similarity < 0.30:
            reasons.append("low intent confidence")
        if best_similarity < self.min_similarity:
            reasons.append("weak historical match")
        if RISK_TERMS.search(message):
            reasons.append("security or high-risk language")
        if intent in {"security_issue", "complaint", "other"}:
            reasons.append("requires human judgment")
        rounded_confidence = round(confidence, 3) if confidence is not None else None
        if reasons:
            return Prediction(
                intent,
                rounded_confidence,
                intent_source,
                None,
                "ESCALATE",
                "; ".join(reasons),
                evidence,
            )
        reply = self._draft_reply(intent, evidence)
        return Prediction(
            intent,
            rounded_confidence,
            intent_source,
            reply,
            "AUTO_HANDLE",
            "relevant historical evidence and no risk trigger",
            evidence,
        )

    @staticmethod
    def _draft_reply(intent: str, evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            return "Thanks for reaching out. A support specialist will review this request."
        reply = str(evidence[0]["agent_reply"])
        reply = re.sub(r"@\w+\s*", "", reply)
        reply = re.sub(r"\s*\^\w{1,4}\s*$", "", reply)
        reply = re.sub(r"\s+", " ", reply).strip()
        return reply


def load_conversations(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"conversation_id", "customer_message", "agent_reply", "intent"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame.dropna(subset=sorted(required))


def run(
    message: str,
    data_path: str | None = None,
    config_path: str = "config.yaml",
) -> dict[str, Any]:
    settings = load_settings(config_path)
    agent = SupportAgent(
        load_conversations(data_path or settings.conversations_path),
        min_confidence=settings.min_intent_confidence,
        min_similarity=settings.min_retrieval_similarity,
        top_k=settings.top_k,
    )
    return asdict(agent.predict(message))


def main() -> None:
    parser = argparse.ArgumentParser(description="Brand-specific support agent")
    parser.add_argument("--message", required=True)
    parser.add_argument("--data")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.message, args.data, args.config), indent=2))


if __name__ == "__main__":
    main()
