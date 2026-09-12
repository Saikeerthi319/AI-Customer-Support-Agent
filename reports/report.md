# BrandSupport AI Report

## 1. Problem framing

This project builds a focused support decision system for one brand. Good means: intent predictions are useful across classes, replies reflect historically observed resolutions, unsupported claims are avoided, and risky cases are routed to humans. It does not attempt live social posting, identity verification, policy lookup, or autonomous handling of security, legal, or payment disputes.

## 2. Data and method

Profile the Customer Support on Twitter export, choose one brand with a documented scoring rule, reconstruct conversation threads, and split by thread ID. Label 150-250 held-out examples across common, rare, ambiguous, and risky cases. The agent uses TF-IDF/logistic regression for intents, TF-IDF cosine retrieval for historical evidence, a deterministic evidence-backed draft in the reproducible baseline, and an explicit escalation gate.

## 3. Baselines and results

Run `python -m evaluation.run` after replacing the fixture. Report majority-class macro F1, TF-IDF/logistic regression macro F1, agent macro F1, reply rubric scores, auto-handling rate, correct auto-handling rate, and unsafe auto-handling rate. Never copy sample numbers into the final report.

## 4. Failure analysis

Document five real examples. Expected categories include ambiguous short messages, rare intents, weak retrieval, conflicting historical resolutions, and unsupported policy claims. For each example include the message, prediction, evidence, consequence, and a hypothesis for the next improvement.

## 5. What is misleading about my headline number?

A safe auto-handling rate is measured on a curated, relatively small golden set and may not match live traffic. Repeated issue phrasing can make retrieval look better than it will under distribution shift. Historical agent replies may contain mistakes, labels are subjective, thresholds can be tuned to this set, and LLM-judge scores are not human ground truth. Report coverage and errors alongside any headline rate.

## 6. One more week

Add hybrid BM25 plus embedding retrieval, conversation context, calibrated probabilities, multiple human annotators, policy-aware claim checking, a larger golden set, and shadow-mode monitoring with drift and escalation audits.
