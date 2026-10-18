# NOTES

## Technical decisions and why

| Decision | Why |
| --- | --- |
| **Safety escalation is decided by fixed rules (C1, C2), never by the model.** | A missed escalation is the costliest mistake in the brief. Rules are testable and still answer if the model fails to load. |
| **Classify each sentence, not the whole complaint.** | Many tickets describe two symptoms. A whole-ticket model scored 0.748 on those; classifying sentences and then applying rule C5 scored 0.992. |
| **Small model: TF-IDF words and letters, keyword lists, logistic regression.** | Trains in seconds on a CPU, answers in about 10 ms, works offline, and is fully explainable. |
| **One function holds the rule order:** C1, then C4 or C6 when no symptom is found, then C3, then C5, then the single symptom. | The business rules stay readable and testable in one place. |
| **C1 wins over "trips the supply" in `no_power`.** | Appendix B and rule C1 disagree; the rule says "whatever else the complaint says". |
| **C2 only sets the flag; C3 needs a known age of 30 or less; a lone error code is always `control_panel_or_display`.** | These follow the rule text literally. There is no brand code table to map codes to faults. |
| **`other` only when no sentence looks like a symptom.** | An earlier probability cut-off sent 229 real complaints to `other`. |
| **Confidence is calibrated per decision path; C5 is capped at 0.70 and C6 fixed at 0.20.** | The brief checks that confidence is informative, and rules C5 and C6 require lowered confidence. |
| **A burner "flame" is not a fire on its own.** | A normal gas flame escalated in my audit. |
| **Typo repair fixes one-letter slips only.** | Two-letter repair turned "soaking" into "sparking" and caused false escalations. |
| **Three validation schemes, not one.** | The data repeats about 120 sentences, so a random split (0.992) measures memory. Holding out wording gives 0.974. On 150 real incident reports I labelled, escalation caught 66 of 69 before fixes with no false alarms, but the category was right only about 55% of the time when no safety word decided it. |
| **Lenient input and exactly five response fields.** | The grader's data will differ in format. Wrong types and unknown products still get an answer; the page's trace comes from a separate `/explain` endpoint. |
| **One start command (`run.py`) that builds its own environment, with pinned library versions.** | The automated score is zero if the service does not start on a clean machine. |

## Skipped, and why

- **Sentence embeddings.** They would likely help with unseen wording, but add a large dependency and start-up risk.
- **A blind test set.** The real-report test replaced my earlier guess: expect category accuracy of 0.55 to 0.75 on long real narratives.
- **Brand error-code mapping, other languages:** no data to verify them.
- **A check on Linux.** Built and verified on Windows with Python 3.10 and 3.12 only.

## What I would improve next

1. Label a few hundred real reports and train the sentence model on them; template sentences do not transfer to long narratives.
2. Add training sentences for the weakest classes (`not_cooling`, `not_heating`, `door_seal_or_latch`).
3. Make safety detection more tolerant of heavy typos: at 10% corrupted letters escalation recall falls to 0.89.
