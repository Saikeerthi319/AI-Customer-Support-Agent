# AmazonHelp Support Agent — Evaluation Report

## 1. Problem framing and scope

The selected brand is **AmazonHelp**. A good result must classify diverse, multilingual tweets, use a historically similar AmazonHelp response as evidence, and avoid autonomously answering ambiguous or risky requests. The system does not post to Twitter, verify identity, look up orders, or resolve security, legal, payment, AWS, or complaint cases.

The intended trust target is high auto-handle precision with escalation recall reported alongside coverage. Fluent wording alone is not success.

## 2. Data and method

The source is the Customer Support on Twitter Kaggle dataset. A seeded sample (`seed=7`) contains 200 unique AmazonHelp customer/reply pairs. A deterministic, intent-aware split (`seed=23`) assigns 50 conversations to development and 150 to the frozen golden set; conversation overlap is zero.

Labels received a full second-pass review against `data/labeling_guide.md` (25 rows corrected). Status is `second_pass_reviewed`. This is independent rubric review, not a separate hired human annotator, and should not be described as candidate-only hand-labeling if that is not how the work was done.

The agent combines:

1. deterministic multilingual keyword rules for high-signal intents;
2. TF-IDF/logistic regression as fallback classification;
3. TF-IDF cosine retrieval over training conversations;
4. a reply copied from the highest-scoring historical response, with handles/signatures removed;
5. escalation for weak evidence (`similarity < 0.45`), low confidence, risky language, complaints, and `other`.

Thresholds and paths are loaded from `config.yaml`. Rule matches expose their source and return no probability rather than presenting an uncalibrated rule score as confidence.

## 3. Results versus baselines

Results from `python -m evaluation.run`:

- Majority-class intent baseline: **3.8% macro F1**
- TF-IDF/logistic-regression baseline: **8.7% macro F1**
- Rule-assisted agent: **33.7% macro F1** and **44.0% accuracy**
- Auto-handle coverage: **1.3%**
- Auto-handle precision: **0.0%** (both auto-handled examples should have escalated)
- Unsafe auto-handle rate: **1.3%**
- Escalation recall: **98.2%**

The agent beats both intent baselines, but it is not trustworthy enough for autonomous deployment: intent macro F1 is low and almost all traffic is escalated. The two auto-handled cases were unsafe.

Independent vs LLM-judge reply scores (`n=40`, `gpt-4o-mini`): aggregate Spearman **ρ = 0.314** (`p = 0.0485`). Correctness ρ = 0.607 and helpfulness ρ = 0.523. Safety Spearman is undefined because the independent rater scored every sampled output 5 for safety: every sampled system output was an escalation. The largest disagreement is that the independent rater treats a correct escalation as moderately correct (4–5), while the LLM judge often scores the same terse `[ESCALATED]` notice as 2–3 because it does not answer the customer.

## 4. Top five failure modes

1. **Non-English risk language.** A Japanese phishing complaint (ID 4) was predicted as `delivery_issue`; retrieval similarity was 0.0. Routing still escalated, limiting harm. Hypothesis: language detection plus multilingual embeddings would recognize both topic and risk.
2. **Order status versus delivery issue.** “when do I get the PS4 I ordered” (ID 12) was labelled `order_status` but predicted `delivery_issue`. Hypothesis: these intents overlap operationally and need either clearer annotation boundaries or a hierarchy with `order_status` as a subtype.
3. **Out-of-domain Amazon services.** An AWS domain/S3 recovery request (ID 3) was predicted `delivery_issue`. Hypothesis: add explicit out-of-domain detection and route Amazon retail versus AWS before intent classification.
4. **Sparse technical vocabulary.** A modem service/RMA question (ID 6) became `other`. Hypothesis: 50 training examples do not cover product-specific technical language; more stratified training data is needed.
5. **Over-escalation of routine asks.** “when do I get the PS4 I ordered” (ID 12) is labelled `AUTO_HANDLE` but was escalated for a weak historical match. Thanks and membership FAQs in the 40-row judge sample show the same pattern. Hypothesis: require a higher similarity bar only for risky intents, and allow high-precision FAQ/thanks templates without retrieval.

## 5. What is misleading about my headline number?

The 33.7% macro F1 improvement sounds large mainly because both baselines are extremely weak on only 50 training examples. Labels are second-pass reviewed, not a multi-annotator gold standard. The set is a seeded convenience sample rather than live traffic. The 98.2% escalation recall is achieved by escalating 98.7% of traffic, so it does not demonstrate useful automation. Auto-handle precision is 0% on two examples. Historical replies can themselves be wrong. The 0.314 judge agreement is statistically weak and inflated/deflated by a sample where every system output was an escalation, so safety has no rank variation.

## 6. What I would do with one more week

Obtain several hundred separate training conversations; merge overlapping intents; add language and out-of-domain detection; use multilingual hybrid retrieval; calibrate routing on development data only; add policy-aware claim checking; double-label a disagreement subset with a second person; then run shadow-mode audits for unsafe automation and drift.
