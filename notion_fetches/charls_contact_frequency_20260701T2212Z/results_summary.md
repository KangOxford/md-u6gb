# Contact Frequency Validation Results

## Sample

| Contact variable | Person-waves | IDs | Share weekly=1 | Within-ID SD |
| --- | ---: | ---: | ---: | ---: |
| `weekly_inperson = h?kcntf` | 9,721 | 3,581 | 0.848 | 0.215 |
| `weekly_any = h?kcnt` | 9,721 | 3,581 | 0.949 | 0.149 |
| `weekly_phone_email = h?kcntpm` | 7,054 | 2,960 | 0.580 | 0.305 |

## Main Effects

| Term | Outcome | Coef | SE | p | N | IDs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `weekly_inperson` | `Q_equal_fixed` | 0.002682 | 0.003802 | 0.481 | 8,764 | 3,219 |
| `weekly_any` | `Q_equal_fixed` | 0.006370 | 0.005765 | 0.269 | 8,764 | 3,219 |
| `weekly_phone_email` | `Q_equal_fixed` | -0.001447 | 0.003190 | 0.650 | 6,323 | 2,651 |

## Competition Table

| Model | `ln_transfer` | `weekly_inperson` | `ln_pension` |
| --- | ---: | ---: | ---: |
| A: transfer + controls | 0.000339, p=0.283 | NA | NA |
| B: A + weekly_inperson | 0.000339, p=0.283 | 0.003440, p=0.368 | NA |
| C: B + pension | 0.000235, p=0.452 | 0.002682, p=0.481 | 0.002881, p<0.001 |

## Robustness

| Check | Result |
| --- | --- |
| Lagged contact | `L1_weekly_inperson -> Q_equal_fixed`: coef `0.002846`, p `0.595` |
| First stage | `child_near -> weekly_inperson`: coef `0.2141`, p `2.3e-34` |
| Reduced form | `child_near -> Q_equal_fixed`: coef `0.00191`, p `0.622` |
| CES-D channel | `weekly_inperson -> cesd10`: coef `-0.0301`, p `0.898` |
| Loneliness channel | `weekly_inperson -> loneliness`: coef `-0.0079`, p `0.846` |
| Life satisfaction channel | `weekly_inperson -> life_satisfaction`: coef `-0.0099`, p `0.774` |

## Decision

The minimum validation does not support promoting weekly in-person contact to a core explanatory variable. The safer paper path is to keep pension/cash-flow as the main line and use contact as an upgraded heterogeneity or family-support-gradient robustness layer.

