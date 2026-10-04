# NCERT Multilingual Math — release 0.2.0

Built 2026-08-13 by `make_release.py`.

## Files

| File | Items | What it is |
|---|---|---|
| `main.jsonl` | 267 | Scorable items whose answer field was normalised deterministically, plus solver-verified perturbations |
| `needs_review.jsonl` | 148 | Answer shape the deterministic layer could not type confidently. NOT release quality. |
| `non_scorable.jsonl` | 46 | Proof / open-ended items with no single scorable answer |

Every item appears twice, once per language, sharing `item_id`.

## What has and has not been done

Done:
- Notation canonicalised to one convention across EN and HI
- EN/HI pairing verified; sub-part text collisions quarantined
- Answer typed, unit extracted into its own field, tolerance set
- `answer_type` assigned once per item from the English row, applied to both languages

NOT done in this build — read before citing:
- **No independent answer verification.** Answers come from the source scrape and
  were normalised, not re-solved. `review_status` never claims otherwise.
- **entity_swap_applied is false for every item.** No decontamination has been
  applied. Items are lexically identical to the public NCERT text.
- Perturbation covers 10 items only, from four
  deterministic templates. All were re-solved independently with SymPy.

## Fields

`review_status` is the field to filter on:
- `auto_normalised` — deterministic typing succeeded
- `perturbed_solver_verified` — perturbed and re-solved from scratch
- `needs_llm_normalisation` — routed to an agent that has not run
