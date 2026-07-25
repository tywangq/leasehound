# Examples

`sample_lease.md` is a **synthetic** lease used as the scanner's acceptance test. Seven clauses deliberately violate the Residential Landlord-Tenant Act; a correct scan flags all seven red and leaves the rest alone.

| Clause | Planted violation | Governing law |
| --- | --- | --- |
| 3 — late charges from day 1 | Late fees within the 5-day grace period are prohibited | RCW 59.18.230(2)(i) |
| 5 — tenant pays landlord's attorneys' fees regardless of outcome | Prohibited provision | RCW 59.18.230(2)(e) |
| 7 — entry at any time without notice | Landlord must give notice before entry | RCW 59.18.150 |
| 9 — NDA on rent amount and lease terms | Prohibited provision | RCW 59.18.230(2)(c) |
| 11 — waiver of chapter rights + class-action ban | Prohibited provisions | RCW 59.18.230(2)(a), (2)(b) |
| 13 — electronic payment only | Tenant cannot be required to pay rent by electronic means only | RCW 59.18.230(2)(j) |
| 14 — landlord exculpation + indemnification | Prohibited provision | RCW 59.18.230(2)(f) |

Clauses 1, 2, 4, 6, 8, 10, 12 are ordinary terms and should come back green (or at most yellow).

The scan also runs a **missing-protections check** against a hand-curated, statute-cited checklist. On the sample it reports two genuine absences — fire safety information and mold information (both RCW 59.18.060) — while the deposit-related items (RCW 59.18.260/.270) correctly come back as addressed.

Run:

```bash
python -m leasehound.scan examples/sample_lease.md
```
