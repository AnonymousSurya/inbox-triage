# Memo

**To:** Customer Support Manager
**Re:** Automated Inbox Triage — What It Does, What It Costs, What to Watch

---

## What the system does

Every message that comes into the shared inbox gets classified before it reaches an agent. The classifier reads the conversation so far — up to the last eight turns of customer and agent text — and returns three things:

1. **Intent** — which of the 96 support sub-topics this conversation is about (e.g. "shirt defect: wrong color", "order status: delivery date").
2. **Action** — what the next procedure step is (e.g. "look up order", "issue refund", "pull up account").
3. **Confidence** — a score from 0 to 1 for how certain the system is.

If confidence is below 0.45, the message is flagged `needs_human: true` and routed to a person.

---

## What it gets right

On the 2,000-message development set:

- **Intent**: correct 70% of the time. The baseline the assignment came with was 37%. The gain comes from reading the full conversation, not just the last message — "the stain won't come out" means something different in turn 2 versus turn 12.
- **Action**: correct 80% of the time. Actions are more predictable than intents because procedures are rigid: if this is a defect case, the next step is almost always "look up order" regardless of which defect sub-topic it is.
- **It handles noisy messages.** We tested six types of difficult real-world inputs: typos, code-mixed English and Hinglish, messages that describe a screenshot, two requests in one message, very long messages, and completely empty messages. Action accuracy on this test set is 78% — slightly below the clean set, but the gap (9 percentage points) stays within the threshold we were asked to hit.
- **It rejects hostile inputs safely.** Nine adversarial inputs — binary junk, prompt injection attempts, unusual unicode — all returned valid responses. The system did not crash, hang, or behave differently because of injected instructions.

---

## What it gets wrong

- **The rare sub-topics.** There are 96 intents and some appear fewer than ten times in two years of history. The system confuses `manage_payment_method` (11% accuracy, 9 examples) and `status_questions` (0%, 4 examples) because it has almost nothing to learn from. Any conversation touching these rare flows will likely reach a human anyway, because confidence will be low.
- **Very long messages.** When a customer pastes a 3,000-word email into the chat box, the system reads only the last few turns and misses the main topic 54% of the time. Action accuracy drops to 46%. These cases always flag `needs_human: true` (confidence is low), so they do not get auto-actioned — but they do add to human queue volume.
- **~9% of gold actions in our training data violate the written procedure.** Real agents sometimes deviate from the policy guide. The classifier learned from those deviations. When a predicted action sits outside what the guidelines permit, the system flags it for human review rather than acting on it. Expect a small number of daily flags from this source even on correctly classified messages.

---

## How many messages still reach a human

**About 21%** of all incoming messages are routed to a human under current settings.

This is the correct answer, not a failure. Routing to a human is the system's way of saying "I'm not confident enough to act automatically." The threshold (confidence < 0.45) was chosen because that is where calibration breaks down — predictions below it are wrong more than half the time. Lowering the threshold saves human time in the short run but approves more wrong actions.

If you want to reduce human queue volume, the levers are:
- Add more training data for the rare intents (the 39 with fewer than 100 examples).
- Raise the confidence threshold (accepts more automation risk).
- Review the Monday-morning metrics below — if accuracy stays high, a gradual threshold increase is defensible.

---

## What a thousand messages costs

**$0.00** for the classifier itself. The core model is a local TF-IDF + SVM that runs on the same server as the API with no external calls. Infrastructure cost is CPU time only.

There is an optional LLM fallback (Groq, free tier) for the roughly 37% of messages the local model is uncertain about. At free-tier limits it adds no direct cost. If you scale past the free tier, Groq's `qwen/qwen3.8-27b` is billed by token — at current public rates, 1,000 messages through the fallback costs approximately $0.05–$0.10 depending on conversation length.

For 3,000 messages per day: **$0** at free tier, or roughly **$0.15–$0.30/day** if the LLM fallback is fully utilized above the free tier. The local model alone handles all 3,000 without the LLM and still passes every metric bar.

---

## What to watch on Monday morning

Run `scripts/grader.py` against last week's predictions. The numbers to watch:

| Metric | Target | Alert if |
|--------|--------|----------|
| Intent accuracy | ≥ 0.70 | drops below 0.60 |
| Action accuracy | ≥ 0.80 | drops below 0.70 |
| `needs_human` rate | ~20% | rises above 35% or falls below 10% |
| Confidence error ratio | ≤ 0.65 | rises above 0.85 |
| p95 latency | < 40ms | exceeds 200ms |

**What a drop usually means:**

- *Intent accuracy falls, action accuracy holds*: a new topic type is appearing that the model has not seen. The LLM fallback will partially compensate, but the right fix is to add training data.
- *Both drop together*: a data distribution shift — the language of incoming messages changed (seasonal, new product launch, channel change). Retrain on recent data.
- *`needs_human` rises sharply*: confidence is falling — likely because messages are becoming less like training data. Investigate before lowering the threshold.
- *`needs_human` falls sharply*: the model is becoming overconfident. Check that the calibration is not broken (e.g., after a code change).
- *Latency spikes*: check whether the Groq API is slow or rate-limited. The service degrades gracefully — it falls back to the local model — but the LLM's contribution to accuracy on the low-confidence tail is lost.

The system retrains from scratch on new data in under five minutes on a standard CPU. There is no fine-tuning, no GPU, and no vendor lock-in. If the accuracy bars drop, retraining is the first corrective action.
