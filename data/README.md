# Data

| Folder | File | What it is |
| --- | --- | --- |
| `training/` | `tickets.csv` | 1,000 labelled tickets the model is trained on |
| | `seed_clauses.csv` | 828 hand-worded sentences, one symptom each, used as extra training data |
| | `make_seed_clauses.py` | Writes `seed_clauses.csv`; edit the sentences here |
| `testing/` | `safety_cases.py` | Safety edge cases used only by the tests, never for training |
| | `real_reports/` | Two hand-labelled samples of real incident reports, with the scripts that score them |
| `samples/` | `sample_tickets.csv`, `sample_tickets.json` | Twelve tickets to try in the Bulk upload tab |
| | `sample_tickets_labelled.csv` | The same twelve with expected answers, so Bulk upload shows a score |

Sources, label policy and results are described in `DATA.md` at the top of the repository.
