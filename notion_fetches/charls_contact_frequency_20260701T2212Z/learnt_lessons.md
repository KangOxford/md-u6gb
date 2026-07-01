# Learnt Lessons

## Contact-variable interpretation

- `h?kcntf` is binary weekly in-person contact, not number of visits per week. It should not be described as a continuous visit-frequency variable.
- `h?kcnt` is combined in-person/phone/email contact and is too saturated in this sample (`weekly=1` share about `0.949`) to carry much within-person identifying variation.
- A strong first stage from `child_near` to contact is not enough for the core-variable claim when the reduced-form effect on `Q_equal_fixed` is near zero.

## Notion reporting

- Markdown alignment rows can render as literal table rows in Notion child pages. For polished result tables, use `<table header-row="true">` tags rather than pipe tables with `---`.
- When a Notion page already contains bracketed instructions, update the exact instruction location first, then write local records.
- If root record files are already dirty, use a task-specific record directory and force-add only the new files rather than staging unrelated root record changes.

