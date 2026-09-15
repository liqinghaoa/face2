# KM-BIO-v2R.1 R-C0R resumed candidate ladder

- Status: `R_C0R_COMPLETE`
- Historical candidates reused read-only: `V2R-0`, `V2R-P`
- Newly executed candidates: `V2R-PS`, `V2R-PSG`
- Selected candidate: `V2R-PS`
- Frozen fold SHA-256: `e763a9f3dc7f58babe264f7056699c3d5b3722187fbb558f83f2fdb5d6b67370`

| Candidate | Median logRMSE | P90 logRMSE | Median RMSE | Median SAM deg | Better than reference | Median model/reference ratio | Strong targets |
|---|---:|---:|---:|---:|---:|---:|---|
| V2R-0 | 0.109281 | 0.199477 | 0.033329 | 5.532 | 0.636 | 0.781 | False |
| V2R-P | 0.096482 | 0.194319 | 0.030363 | 4.894 | 0.682 | 0.680 | False |
| V2R-PS | 0.061478 | 0.116877 | 0.018109 | 3.484 | 0.932 | 0.443 | False |
| V2R-PSG | 0.058819 | 0.110759 | 0.017952 | 3.297 | 0.909 | 0.433 | False |

Global-profile width and boundary flags are reported in `global_profile_summary.csv`; they are diagnostic under v2R.1 and did not stop PSG.
