# CortexHR attrition score: one-page note for managers

**What the score is.** For each employee the model estimates how likely they are to leave in the next 12 months, based on pay position, raises, engagement, workload, tenure and recent attendance. We rank everyone and group them: **High** = top 10% of scores, **Medium** = next 20%, **Low** = the remaining 70%.

**How well it works (held-out data, 477 employees, 70 actual leavers).**
- Overall 14.7% left. In the **High** band 38% left (18 of 48), versus 9% in the **Low** band.
- The High band holds **26%** of all leavers (random picking would give 10%); High + Medium hold 56%.
- Ranking quality is moderate: ROC-AUC 0.68, PR-AUC 0.32. It is useful for prioritising conversations, not for predicting individuals.

**What the score is NOT.**
- Not a decision. It does not know why someone might leave, and most High-band people (62%) did **not** leave.
- Not a probability you can quote: raw scores are inflated by design (0.40 average vs 0.147 actual). Use the band, not the number.
- Not evidence about performance, conduct or potential. Never use it to deny a raise, promotion or role, or to start any disciplinary step.

**Limitations.** Small number of leavers; 84% of rows lack attendance/payroll history; people who had already resigned are partly included; fairness across groups has not yet been tested (protected attributes are held separately for that audit).

**How to use it.** (1) A person, not the model, reviews each High-band name. (2) Ask what could be fixed: pay gap, stalled raise, manager change, workload. (3) Combine with what you know first-hand; ignore the score when it conflicts with facts. (4) Record the human decision, not the score. (5) Re-check monthly: scores go stale.

*Decision support only. The model must never independently determine an employment action.*
