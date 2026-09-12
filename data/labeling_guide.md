# AmazonHelp Annotation Guide

## Goal

Create two conversation-disjoint files from the prepared AmazonHelp pairs:

- `data/annotation_train.csv`: labeled examples used to develop the classifier (50 in the checked-in sample; 200-400 preferred).
- `data/golden_set.csv`: 150-250 labeled examples held out until final evaluation.

Never use the same `conversation_id` in both files.

## Intent taxonomy

Use one primary intent per customer message:

- `delivery_issue`: late, missing, damaged, or tracking questions about delivery.
- `order_status`: where an order is, confirmation, or general order progress.
- `refund_return`: refund, return, reimbursement, or money-back request.
- `payment_issue`: charge, payment failure, duplicate charge, or billing concern.
- `account_access`: login, password, account settings, or account recovery.
- `cancellation_change`: cancel, modify, or change an order.
- `product_question`: product availability, compatibility, features, or product information.
- `technical_issue`: website, app, device, checkout, or technical error.
- `complaint`: anger, dissatisfaction, request for manager, or service complaint.
- `other`: messages that do not fit the taxonomy or are too ambiguous.

Choose the customer’s main requested outcome. If two intents are equally plausible, use `other` and explain the ambiguity in `expected_reason`.

## Expected action

Use `AUTO_HANDLE` only when the message is a routine request with clear historical evidence and no sensitive risk. Use `ESCALATE` for:

- account takeover, fraud, unauthorized access, threats, legal claims, or safety concerns;
- angry complaints requiring judgment or a manager;
- ambiguous messages with insufficient context;
- cases where historical replies conflict or no useful historical match exists.

## Labeling process

1. Read the customer message and the paired historical reply.
2. Assign exactly one intent from the list above.
3. Assign `AUTO_HANDLE` or `ESCALATE` using the rules above.
4. Write a short reason, especially for escalation.
5. Re-read all labels once after the first pass.
6. Review the AI-drafted annotation batches and correct every disagreement.
7. Run `python -m src.build_datasets` to make the deterministic split.
8. Confirm `label_status=second_pass_reviewed` after a full re-read, or `human_verified` only if a person independently confirmed every row.

The source `data/golden_set_amazonhelp.csv` is an unlabeled 200-row sample. The checked-in annotation batches are drafts, not proof of candidate hand-labeling.

## Required columns

`id, conversation_id, customer_message, agent_reply, intent, expected_action, expected_reason`

Do not submit final metrics until every row is complete and human-verified.
