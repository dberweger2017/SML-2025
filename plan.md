# SML Project 1: Progress Summary (Updated after 30x30 RGB Run)

## 1. Summary of Learnings So Far:

*   **Model Viability:** `HistGradientBoostingRegressor` combined with `StandardScaler` and `PCA` remains the best approach found so far.
*   **Best Performance:** The best performance achieved is now **~17.16 cm MAE (Cross-Validation)** and **~16.92 cm MAE (Validation Set)**.
*   **Data Representation:**
    *   Using 60x60 images (`downsample_factor: 5`), both RGB and Grayscale, hit a performance plateau around ~17.3 cm MAE. Grayscale offered no advantage over RGB at this resolution.
    *   **Switching to 30x30 RGB images (`downsample_factor: 10`) provided a noticeable improvement**, breaking the previous plateau and achieving the current best scores.
*   **Dimensionality Reduction (PCA):** Aggressive PCA remains **essential**.
    *   For 60x60 RGB, ~30 components were optimal.
    *   For 60x60 Grayscale, ~20 components were optimal.
    *   For **30x30 RGB**, ~**20 components** were optimal (out of 2700 initial features).
    *   This consistently suggests that only a very low-dimensional subspace captured by PCA is useful for this model.
*   **Training Iterations (`max_iter`):** The model **consistently performs best at the maximum number of iterations** allowed in the grid search across all tested resolutions (up to 2000 iterations in the last successful run). This strongly indicates the model benefits from long training times and might still be slightly undertrained even at 2000 iterations.
*   **Learning Rate (`learning_rate`):** A low learning rate (**0.02**) consistently performs best when paired with high iterations.
*   **Tree Complexity (`max_leaf_nodes`):** Moderate complexity (**50**) seems slightly favored over lower complexity (31), but the difference might be small.
*   **Progress:** Initial gains were followed by a plateau at ~17.3 cm using 60x60 images. **Changing resolution to 30x30 RGB successfully broke this plateau**, leading to the current best MAE around ~17.0 cm.

## 2. What We Know About the Dataset/Problem Now:

*   The relationship between raw pixels and distance is complex. Reducing spatial resolution (from 60x60 to 30x30) seems beneficial for the HGBR+PCA model, potentially reducing noise or focusing the model on more relevant coarse features.
*   Regardless of resolution (60x60 or 30x30) or color (RGB vs Grayscale), the **essential information for the HGBR model seems concentrated in an extremely low-dimensional subspace** (captured by the top ~20-30 PCA components).
*   Simply adding more PCA components beyond this optimal low number consistently hurts performance within the tested training limits.
*   The `HistGradientBoostingRegressor` model requires **many iterations (2000+)** to perform optimally on this data, especially with the preferred low learning rate (0.02).
*   At 60x60 resolution, color information didn't hurt, but its removal didn't help either. The current best result uses 30x30 RGB.

## 3. Comparison with Previous Summary (Changes Highlighted):

*   **Model Viability:** Confirmed, performance improved to **~17.0 cm MAE**.
*   **Data Representation:** **Finding:** **30x30 RGB (`downsample_factor: 10`) is superior** to 60x60 (RGB or Grayscale) for HGBR+PCA. The previous plateau was broken.
*   **Dimensionality Reduction (PCA):** Principle confirmed (aggressive PCA is key). Optimal components now noted as **~20 for 30x30 RGB**.
*   **Training Iterations (`max_iter`):** Principle confirmed. Model **still maxed out iterations (at 2000)** even with lower resolution input.
*   **Learning Rate (`learning_rate`):** Principle confirmed (0.02 best).
*   **Tree Complexity (`max_leaf_nodes`):** Principle confirmed (50 slightly favored).
*   **Progress/Plateau:** **Finding:** The previous plateau **was broken** by changing the resolution. New best MAE achieved.
*   **Updated "What We Know":** Added the finding that **lower resolution (30x30) helps** this specific model setup. Reinforced the low-dimensional nature and need for high iterations.