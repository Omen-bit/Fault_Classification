# AI tools used

I used AI and research tools during this assignment. This file explains what I used each tool for and how I checked the results.

## Tools

| Tool | Used for |
|---|---|
| Perplexity | Researching background information and possible approaches for the assignment. |
| Google AI Search | Finding information to support my research and compare possible approaches. |
| Antigravity | Assisting with building and debugging the application. |
| Claude (Anthropic) | Assisting with the dataset sourced from Kaggle, including its preparation and analysis. |

## How I used the tool

### Understanding the problem and researching solutions

- I used Perplexity and Google AI Search to research possible approaches and compare design choices.
- I checked the proposed design against the assignment brief, including its safety rules, evaluation requirements, and startup requirements.

### Dataset

- I sourced the dataset from Kaggle and used Claude to assist with preparing and analysing it.
- Dataset sources, preparation steps, and limitations are described in `DATA.md`.

### Building and debugging

- I used Antigravity to assist with building and debugging the application, including the classifier, training and evaluation code, API, start script, and web page.
- The classifier design changed during development to better handle tickets containing multiple symptoms.
- I checked and corrected issues found during development, including safety phrase handling, slow processing on long inputs, and page display problems.

### Documentation

- AI tools assisted with drafting and revising project documentation, including `README.md`, `DATA.md`, `NOTES.md`, and this file.
- I reviewed the documentation against the project and assignment requirements.

## How I tested and verified the results

- **Automated tests:** 401 tests were run offline with outbound connections blocked. They cover the brief’s examples, response format, unusual input, business rules, safety cases, and the page.
- **Evaluation gates:** `python -m fault.evaluate` checks 13 limits and fails if any is missed.
- **Clean-start check:** `python scripts/verify_clean.py` copies the repository to an empty folder, runs the documented start command, and sends the brief’s examples. It passed on Python 3.10 and 3.12.
- **Fresh-ticket checks:** Two sets of tickets with new wording were evaluated. The first set had 70 tickets, with 47 matching the expected answer; the second had 40 tickets, with 34 matching. The sets and limitations are described in `DATA.md`.
- **Real incident reports:** The service was run against 19,856 incident reports. Two samples, of 100 and 50 reports, were manually labelled before reviewing the predictions. The results are described in `DATA.md`.
- **Load check:** The service was checked under bursts of requests. The check exposed blocking during classification, which was corrected and measured again.
- **Manual review:** I used the page for single-ticket entry, bulk upload, and the dashboard, and reviewed the code so I can explain how it works.

## Limitations

- AI tools helped create some rules, training examples, and test cases. These checks show how the system handles the cases included; they do not prove how it will perform on every unseen complaint.

## Runtime

The running service does not call an AI model or external API. It is designed to work with the network switched off after setup.