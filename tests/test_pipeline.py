from src.pipeline import SupportAgent, load_conversations


def test_security_message_escalates():
    agent = SupportAgent(load_conversations("data/sample_conversations.csv"))
    prediction = agent.predict("Someone hacked my account")
    assert prediction.decision == "ESCALATE"
    assert prediction.reply is None


def test_refund_message_gets_grounded_reply():
    agent = SupportAgent(load_conversations("data/sample_conversations.csv"), min_similarity=0.18)
    prediction = agent.predict("Where is my refund?")
    assert prediction.intent == "refund_return"
    assert prediction.intent_source == "rule"
    assert prediction.intent_confidence is None
    assert prediction.decision == "AUTO_HANDLE"
    assert prediction.reply
    assert prediction.evidence
