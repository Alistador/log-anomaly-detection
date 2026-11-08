Test period: 19.2 h, 362104 blocks, 8521 labelled anomalies.

| detector | budget/day | alerts | alerts/day (test) | true alerts | precision | recall |
|---|---|---|---|---|---|---|
| pca | 20 | 43 | 54 | 39 | 0.907 | 0.0046 |
| pca | 100 | 105 | 131 | 92 | 0.876 | 0.0108 |
| pca | 500 | 295 | 368 | 273 | 0.925 | 0.0320 |
| pca | 2000 | 1143 | 1426 | 773 | 0.676 | 0.0907 |
| pca | 10000 | 4265 | 5320 | 2585 | 0.606 | 0.3034 |
| iforest | 20 | 2 | 2 | 2 | 1.000 | 0.0002 |
| iforest | 100 | 36 | 45 | 23 | 0.639 | 0.0027 |
| iforest | 500 | 36 | 45 | 23 | 0.639 | 0.0027 |
| iforest | 2000 | 36 | 45 | 23 | 0.639 | 0.0027 |
| iforest | 10000 | 6867 | 8566 | 5069 | 0.738 | 0.5949 |

| detector | average precision (test) |
|---|---|
| pca | 0.477 |
| iforest | 0.811 |
