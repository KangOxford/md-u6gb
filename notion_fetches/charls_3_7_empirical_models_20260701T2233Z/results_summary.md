# 3.7 Empirical Model Rerun Summary

## Sample

| Sample | N | IDs | Notes |
| --- | ---: | ---: | --- |
| W1-W4 all | 13,449 | 4,878 | Before model-specific missing drops |
| Low-contact available | 9,721 | 3,581 | Has living child and `h?kcntf` available |
| Low-transfer available | 12,703 | 4,859 | Transfer variable available |

## Main Effects

| Outcome | Term | Coef | SE | p | N | IDs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Q_equal_fixed` | `ln_pension` | 0.002949 | 0.000282 | <0.001 | 11,521 | 4,187 |
| `Q_equal_fixed` | `ln_transfer` | 0.000215 | 0.000271 | 0.427 | 11,521 | 4,187 |

## Mechanism First Stages

| Outcome | `ln_pension` coef | p | `ln_transfer` coef | p |
| --- | ---: | ---: | ---: | ---: |
| `cesd10` | -0.0582 | 0.0008 | 0.0207 | 0.202 |
| `loneliness` | -0.00386 | 0.263 | 0.00491 | 0.122 |
| `life_satisfaction` | 0.00668 | 0.0052 | 0.00232 | 0.346 |
| `adl_basic` | -0.00584 | 0.062 | 0.00480 | 0.128 |

## Mechanism Closure

| Mediator in Q model | Coef | p |
| --- | ---: | ---: |
| `cesd10` | -0.00905 | <0.001 |
| `loneliness` | -0.0267 | <0.001 |
| `life_satisfaction` | 0.0625 | <0.001 |
| `adl_basic` | -0.0399 | <0.001 |

## Heterogeneity

| Moderator | Interaction term | Outcome | Coef | p |
| --- | --- | --- | ---: | ---: |
| `low_contact` | `ln_pension_x_low_contact` | `Q_equal_fixed` | -0.000376 | 0.595 |
| `low_transfer_zero` | `ln_pension_x_low_transfer_zero` | `Q_equal_fixed` | -0.000718 | 0.150 |

## Conclusion

The 3.7 model can support the pension main effect and a CES-D/life-satisfaction psychological channel. It does not support a significant low-contact or low-transfer heterogeneity claim in this rerun.

