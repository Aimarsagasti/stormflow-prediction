# Iter18 - Comparativa del clasificador binario XGBoost

Clasificadores binarios entrenados sobre el mismo split temporal de iter17, con target operacional `max(stormflow[t+1..t+h]) >= U`. El umbral operativo se eligio exclusivamente en validacion para cumplir recall >= 0.85 cuando fue posible.

| Variante | Prevalencia test | AUC-PR | ROC-AUC | Prec@0.5 | Rec@0.5 | F1@0.5 | Prec@op | Rec@op | F1@op | Thr op | Lead time mediano (min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h6_u25 | 0.0041 | 0.5842 | 0.9806 | 0.094 | 0.892 | 0.170 | 0.102 | 0.884 | 0.183 | 0.5479 | 25.0 |
| h6_u50 | 0.0012 | 0.2488 | 0.9678 | 0.043 | 0.708 | 0.081 | 0.007 | 0.949 | 0.014 | 0.0456 | 30.0 |
| h12_u25 | 0.0066 | 0.4180 | 0.9397 | 0.078 | 0.787 | 0.141 | 0.042 | 0.830 | 0.080 | 0.3011 | 60.0 |
| h12_u50 | 0.0021 | 0.2050 | 0.9187 | 0.030 | 0.658 | 0.057 | 0.008 | 0.901 | 0.015 | 0.1049 | 60.0 |

## Criterios de exito orientativos (§8.1)

- h6_u25: precision@op=0.102  recall@op=0.884 -> NO
- h6_u50: precision@op=0.007  recall@op=0.949 -> NO
- h12_u25: precision@op=0.042  recall@op=0.830 -> OK
- h12_u50: precision@op=0.008  recall@op=0.901 -> OK

## Narrativa breve

- h6_u25 (H=6, U=25): AUC-PR=0.5842, precision@op=0.102, recall@op=0.884, lead time mediano=25.0 min.
- h6_u50 (H=6, U=50): AUC-PR=0.2488, precision@op=0.007, recall@op=0.949, lead time mediano=30.0 min.
- h12_u25 (H=12, U=25): AUC-PR=0.4180, precision@op=0.042, recall@op=0.830, lead time mediano=60.0 min.
- h12_u50 (H=12, U=50): AUC-PR=0.2050, precision@op=0.008, recall@op=0.901, lead time mediano=60.0 min.

## Figuras asociadas

- `outputs/figures/iter18/pr_curves.png`
- `outputs/figures/iter18/calibration.png`
- `outputs/figures/iter18/lead_time_distribution.png`