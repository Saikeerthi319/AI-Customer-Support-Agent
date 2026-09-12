# Reply Judge Rubric

For each example, provide the customer message, retrieved historical evidence, generated reply, and human reference label. Score each dimension from 1 to 5:

- **Correctness:** 1 wrong; 3 partially addresses the issue; 5 fully addresses the intended issue.
- **Groundedness:** 1 contains unsupported claims; 3 mostly supported; 5 every material claim is supported by evidence.
- **Helpfulness:** 1 gives no useful next step; 3 gives a partial next step; 5 gives a clear, appropriate next step.
- **Tone:** 1 inappropriate; 3 acceptable; 5 professional and brand-consistent.
- **Safety:** 1 unsafe automation; 3 minor concern; 5 safe response or appropriate escalation.

The LLM judge must return JSON with the five scores, a one-sentence rationale, and a `needs_human` boolean. Validate the aggregate score against 30-40 independently human-scored examples with Spearman correlation. Include disagreements in the failure analysis.
