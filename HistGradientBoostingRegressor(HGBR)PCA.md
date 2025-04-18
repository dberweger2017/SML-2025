## Training Run Summary: HistGradientBoostingRegressor (HGBR) + PCA

| Run | Model | Downsample | RGB   | Best PCA Comps | Best LR | Best Max Iter | Best Max Leaves | Best CV MAE (cm) | Val MAE (cm) | Notes / Purpose                                    |
|-----|-------|------------|-------|----------------|---------|---------------|-----------------|------------------|--------------|----------------------------------------------------|
| 1   | HGBR  | 5 (60x60)  | True  | 50             | 0.1     | 200           | Default         | 18.053           | 18.553       | Initial test, small grid                           |
| 2   | HGBR  | 5 (60x60)  | True  | 30             | 0.05    | 500           | 31              | 17.448           | 17.622       | Expand grid based on Run 1 edges                   |
| 3   | HGBR  | 5 (60x60)  | True  | 30             | 0.02    | 1000          | 50              | 17.317           | 17.251       | Push iterations higher, refine PCA/LR/Leaves       |
| 4   | HGBR  | 5 (60x60)  | True  | 30             | 0.02    | 2000          | 50              | 17.269           | 17.210       | Push iterations even higher (up to 2000)           |
| 5   | HGBR  | 5 (60x60)  | False | 20             | 0.02    | 2000          | 50              | 17.273           | 17.652       | Test grayscale at 60x60                            |
| 6   | HGBR  | 5 (60x60)  | False | 20             | 0.02    | 2000          | 50              | 17.273           | 17.652       | Confirm iter limit for grayscale 60x60 (grid->2500)|
| 7   | HGBR  | 10 (30x30) | True  | 20             | 0.02    | 2000          | 50              | 17.160           | 16.922       | Test lower resolution (30x30 RGB)                  |
| 8   | HGBR  | 10 (30x30) | True  | 20             | 0.02    | 2500          | 50              | 17.159           | 16.917       | Confirm iter limit for 30x30 RGB (grid->3000)      |
| 9   | HGBR  | 20 (15x15) | True  | 30             | 0.02    | 2500          | 50              | 16.937           | 16.754       | Test even lower resolution (15x15 RGB)             |
| 10  | HGBR  | 30 (10x10) | True  | 20             | 0.02    | 3000          | 50              | **16.773**       | **16.700**   | Test lowest resolution (10x10 RGB) - **Best Yet**  |