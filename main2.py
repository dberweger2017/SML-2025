import numpy as np
import joblib  # for saving the final pipeline

from utils import (
    load_config,
    load_dataset,
    load_test_dataset,
    print_results,
    save_results,
)

# --- scikit‑learn imports ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA
from sklearn.experimental import enable_halving_search_cv  # noqa: enables HalvingRandomSearchCV
from sklearn.model_selection import HalvingRandomSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    StackingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV


if __name__ == "__main__":
    # 1. Load configuration
    print("[INFO]: Loading configuration...")
    config = load_config()

    # 2. Load training data
    print("[INFO]: Loading training dataset...")
    X, y = load_dataset(config, split="train")
    print(f"[INFO]: Training data shape: {X.shape}, {y.shape}")

    # 3. Train/Validation split
    print("[INFO]: Splitting data into train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[INFO]: Train/Val shapes: {X_train.shape}, {X_val.shape}")

    # 4. Define stacking ensemble (no SVR)
    print("[INFO]: Setting up stacking regressor...")
    base_learners = [
        (
            'hgb',
            HistGradientBoostingRegressor(
                learning_rate=0.02,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.1,
                tol=1e-3,
                verbose=0,            # reduce per-iteration output
                random_state=42,
            ),
        ),
        (
            'rf',
            RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            'et',
            ExtraTreesRegressor(
                n_estimators=200,
                max_depth=None,
                n_jobs=-1,
                random_state=42,
            ),
        ),
        (
            'knn',
            KNeighborsRegressor(n_neighbors=5),
        ),
    ]

    stack = StackingRegressor(
        estimators=base_learners,
        final_estimator=RidgeCV(),
        n_jobs=-1,
    )

    # 5. Build full pipeline: scaling → RBF features → PCA → stacking
    print("[INFO]: Defining pipeline with RBF + PCA + Stacking...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rbf', RBFSampler(gamma=1e-4, n_components=500, random_state=42)),
        ('pca', PCA(random_state=42)),
        ('stack', stack),
    ])

    # 6. Hyperparameter search via successive halving (exhaustive sampling of 108 combos)
    print("[INFO]: Setting up hyperparameter search (HalvingRandomSearchCV)...")
    param_dist = {
        'pca__n_components': [20, 30, 40, 50],
        'stack__rf__n_estimators': [100, 200, 400],
        'stack__et__n_estimators': [100, 200, 400],
        'stack__knn__n_neighbors': [3, 5, 7],
    }

    neg_mae = make_scorer(mean_absolute_error, greater_is_better=False)
    search = HalvingRandomSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_candidates=108,            # sample *every* combination upfront
        factor=3,
        resource='stack__hgb__max_iter',  # uses HGBR iterations as resource
        max_resources=3000,
        min_resources=100,
        cv=5,
        scoring=neg_mae,
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    # 7. Run search
    print("[INFO]: Starting hyperparameter search...")
    search.fit(X_train, y_train)

    # 8. Report best results
    print(f"[INFO]: Best parameters found: {search.best_params_}")
    best_cv_mae_cm = -search.best_score_ * 100
    print(f"[INFO]: Best CV MAE: {best_cv_mae_cm:.3f} cm")

    best_pipeline = search.best_estimator_

    # 9. Evaluate on validation set
    print("[INFO]: Evaluating on validation set...")
    y_val_pred = best_pipeline.predict(X_val)
    print_results(y_val, y_val_pred)

    # 10. Load and predict test set
    print("[INFO]: Loading test dataset...")
    X_test = np.array(load_test_dataset(config))
    print(f"[INFO]: Test data shape: {X_test.shape}")
    print("[INFO]: Predicting test set...")
    y_test_pred = best_pipeline.predict(X_test)

    # 11. Save predictions and pipeline
    print("[INFO]: Saving predictions to 'prediction.csv'...")
    save_results(y_test_pred)
    print("[INFO]: Saving pipeline to 'pipeline.joblib'...")
    joblib.dump(best_pipeline, 'pipeline.joblib')

    print("[INFO]: --- Script Finished ---")