# SML Project 1: Distance Estimation Game Plan

## 1. The Problem: Simplified

*   **Goal:** Build a system that looks at a single camera image from the ANYmal robot and predicts the distance (in meters) to the nearest object in that image.
*   **Input:** A flattened array of pixel values from an image (either color RGB or grayscale).
*   **Output:** A single floating-point number (predicted distance).
*   **Challenge:** Images have high dimensionality (many pixels/features). The relationship between raw pixel values and distance is likely complex and non-linear. Lighting and object variations add noise.
*   **Constraint:** Cannot use Support Vector Regressors (SVRs).
*   **Toolbox:** Primarily `scikit-learn`.
*   **Measure of Success:** Mean Absolute Error (MAE) between predicted and actual distances, measured in centimeters on a hidden test set. Target: < 8cm.

## 2. The Core Workflow (The Plan)

1.  **Setup:** Ensure your Python environment (local Conda or JupyterHub) is ready with all packages from `requirements.txt`.
2.  **Configure Data Loading:** Modify `config.yaml` to set `load_rgb` and `downsample_factor`. This is your *first* major experimental variable.
3.  **Load Data:** Run `main.py` to load training images and distances using `load_dataset`.
4.  **Split Data (Optional but Recommended):** Use `train_test_split` from `sklearn.model_selection` to create a local validation set (e.g., 80% train, 20% validation) using a fixed `random_state` for reproducibility. This lets you test pipeline performance locally.
5.  **Define Pipeline:** Create a `sklearn.pipeline.Pipeline` object. This should *always* include a scaling step (`StandardScaler`) and *optionally* include a dimensionality reduction step (`PCA`). The final step is the regressor model.
6.  **Define Hyperparameter Grid:** Create a `param_grid` dictionary for use with `GridSearchCV`. This grid should specify the different models and preprocessing parameters you want to test.
7.  **Tune Hyperparameters:** Use `GridSearchCV` from `sklearn.model_selection` with the pipeline, parameter grid, and training data. Use `scoring='neg_mean_absolute_error'` and set `cv` (e.g., 5 folds). Use `n_jobs=-1` to speed up computation.
8.  **Analyze CV Results:** Examine `grid_search.best_score_` (remembering it's negative MAE, so multiply by -100 for cm) and `grid_search.best_params_`. This tells you which combination performed best in cross-validation.
9.  **Evaluate (Optional):** If you created a validation set, use `grid_search.best_estimator_.predict()` on your validation features (`X_val`) and calculate the MAE against `y_val` using `mean_absolute_error` from `sklearn.metrics`.
10. **Train Final Model:** The `grid_search` object with `refit=True` (default) already contains the best pipeline trained on the *entire* dataset it was given. Store this best pipeline using `joblib.dump(grid_search.best_estimator_, 'pipeline.joblib')`.
11. **Predict on Test Set:** Load the test images using `load_test_dataset`. Use `best_pipeline.predict()` to get predictions.
12. **Format and Save:** Save the test predictions to `prediction.csv` using the provided `save_results` function.
13. **Submit:**
    *   Submit `prediction.csv` to Kaggle.
    *   Submit `pipeline.joblib`, the *exact* `config.yaml` used to generate it, and `main.py` (zipped) to Moodle.
14. **Iterate:** Analyze Kaggle public score and CV/validation scores. Go back to step 2 or 5, change your configuration/pipeline/grid, and repeat to improve the score.

## 3. Things to Test Systematically (The Experiment List)

Explore the impact of different choices by testing combinations of the following:

**A. Data Representation (`config.yaml`):**
*   `load_rgb`:
    *   Test `True` (Color)
    *   Test `False` (Grayscale)
*   `downsample_factor`: (Test several values, maybe start broad then refine)
    *   Test `5` (60x60 image)
    *   Test `8` (37x37 image approx)
    *   Test `10` (30x30 image)
    *   Test `12` (25x25 image)
    *   Test `15` (20x20 image)
    *   Test `20` (15x15 image)
    *   *Note:* Smaller factors (<5) might hit memory limits, especially with RGB. Larger factors might lose too much detail.

**B. Preprocessing (within the `Pipeline`):**
*   **Scaling:**
    *   Use `StandardScaler` (Generally the best choice here).
*   **Dimensionality Reduction (`PCA`):**
    *   Test *with* PCA vs. *without* PCA (comment out the PCA step in the pipeline).
    *   If using PCA, test different `n_components` (include in `GridSearchCV` `param_grid`):
        *   Variance-based: `0.99`, `0.95`, `0.90`
        *   Fixed number: `50`, `100`, `150`, `200`, `300` (Adjust based on downsample factor)

**C. Model Choice (Regressor in the `Pipeline`):**
*   Test `HistGradientBoostingRegressor` (Primary candidate)
*   Test `RandomForestRegressor` (Primary candidate)
*   Test `KNeighborsRegressor` (*Strongly recommended to only test this WITH PCA*)
*   Test `Ridge` (As a simple linear baseline)
*   *How to test in GridSearchCV:* Define multiple pipeline configurations in the `param_grid` or run separate `GridSearchCV` instances for different model types.

**D. Hyperparameters (within `GridSearchCV` `param_grid` for each model):**
*   **For `HistGradientBoostingRegressor`:**
    *   `regressor__learning_rate`: `[0.01, 0.05, 0.1, 0.2]`
    *   `regressor__max_iter`: `[100, 200, 300, 500]`
    *   `regressor__max_leaf_nodes`: `[31, 50, 70, 90]`
    *   `regressor__l2_regularization`: `[0.0, 0.1, 1.0]`
*   **For `RandomForestRegressor`:**
    *   `regressor__n_estimators`: `[100, 200, 300, 400]`
    *   `regressor__max_depth`: `[None, 10, 20, 30]`
    *   `regressor__min_samples_split`: `[2, 5, 10]`
    *   `regressor__min_samples_leaf`: `[1, 3, 5]`
    *   `regressor__max_features`: `['sqrt', 'log2', 0.7]`
*   **For `KNeighborsRegressor` (if testing):**
    *   `regressor__n_neighbors`: `[3, 5, 7, 10, 15]`
    *   `regressor__weights`: `['uniform', 'distance']`
    *   `regressor__p`: `[1, 2]` (Manhattan vs Euclidean distance)
*   **For `Ridge` (if testing):**
    *   `regressor__alpha`: `[0.1, 1.0, 10.0, 100.0, 1000.0]`

**E. Cross-Validation:**
*   `cv` parameter in `GridSearchCV`: Test `5` (standard), maybe `3` (faster) or `10` (more robust estimate, slower).

## 4. Key Strategy

*   **Start Broad:** Test different `downsample_factor` values and `load_rgb` settings first, perhaps with default hyperparameters for `HistGradientBoostingRegressor` or `RandomForestRegressor` and basic PCA (e.g., `n_components=100`). Find a promising data representation.
*   **Refine:** Once you have a decent data setup (e.g., grayscale, factor=10), focus `GridSearchCV` on testing PCA variations and detailed hyperparameters for the top 1-2 models (`HistGradientBoostingRegressor`, `RandomForestRegressor`).
*   **Track Everything:** Keep a log (spreadsheet or text file) of your experiments: `config.yaml` settings, PCA settings, model used, hyperparameter grid searched, best CV MAE, validation MAE (if used), and Kaggle public MAE. This prevents repeating work and helps identify trends.