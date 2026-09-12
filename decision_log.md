# Decision Log

1. **One brand:** Focused scope makes intent definitions and reply style measurable.
2. **Conversation-level split:** Prevents leakage between customer and agent turns from the same thread.
3. **Ten intents:** Keeps labels actionable while separating order status from delivery failures.
4. **AmazonHelp:** High volume and diverse public issue types make it a useful stress test.
5. **TF-IDF/logistic regression baseline:** Fast, interpretable, and reproducible.
6. **Historical retrieval:** Replies should reflect observed brand resolutions rather than generic chatbot behavior.
7. **Evidence before generation:** Auto-handling requires a historical match of at least 0.45 cosine similarity.
8. **Explicit escalation:** A wrong confident answer can be worse than a human handoff.
9. **Security and complaints escalate:** These cases need identity, policy, or empathy judgment not available in this MVP.
10. **Macro F1:** Intent classes are likely imbalanced, so accuracy alone would hide rare-class failures.
11. **Conversation-disjoint 50/150 split:** Meets the golden-set minimum without leaking one conversation into training.
12. **LLM judge requires human validation:** Judge scores are measurements, not ground truth.
13. **No frontend in MVP:** Evaluation and reproducibility are higher-value than a demo UI.
14. **Synthetic fixture checked in:** Reviewers can run the pipeline without credentials; it is clearly not final evidence.
15. **Draft labels then second-pass review:** All 200 rows were re-read against the labeling guide; 25 labels were corrected and marked `second_pass_reviewed`.
