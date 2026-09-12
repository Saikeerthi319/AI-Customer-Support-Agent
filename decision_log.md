# Decision Log

1. **One brand:** Focused scope makes intent definitions and reply style measurable.
2. **Conversation-level split:** Prevents leakage between customer and agent turns from the same thread.
3. **8-12 intents target:** Keeps labels actionable and reduces ambiguous fine-grained classes.
4. **Brand selected by profile:** Volume, resolution rate, and intent diversity are explicit selection criteria.
5. **TF-IDF/logistic regression baseline:** Fast, interpretable, and reproducible.
6. **Historical retrieval:** Replies should reflect observed brand resolutions rather than generic chatbot behavior.
7. **Evidence before generation:** No reply should be auto-handled without a relevant historical match.
8. **Explicit escalation:** A wrong confident answer can be worse than a human handoff.
9. **Security and complaints escalate:** These cases need identity, policy, or empathy judgment not available in this MVP.
10. **Macro F1:** Intent classes are likely imbalanced, so accuracy alone would hide rare-class failures.
11. **Golden set is frozen:** Prevents tuning against a moving test target.
12. **LLM judge requires human validation:** Judge scores are measurements, not ground truth.
13. **No frontend in MVP:** Evaluation and reproducibility are higher-value than a demo UI.
14. **Synthetic fixture checked in:** Reviewers can run the pipeline without credentials; it is clearly not final evidence.
15. **Headline metric is safe auto-handling:** Automation quality matters more than raw classification accuracy.
