import numpy as np
import joblib # For saving the pipeline
from utils import (
    load_config,
    load_dataset,
    load_test_dataset,
    print_results, # Use this for validation results
    save_results,
)

# --- Scikit-learn Imports ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# *** REMOVE or COMMENT OUT the HGBR import ***
# from sklearn.ensemble import HistGradientBoostingRegressor
# *** ADD the RandomForestRegressor import ***
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, make_scorer

# --- Configuration ---
# Ensure config.yaml has load_rgb: True and downsample_factor: 30 (or your desired best)


if __name__ == "__main__":
    # 1. Load Configuration from config.yaml
    print("[INFO]: Loading configuration...")
    config = load_config()
    # Verify the settings (optional) - Make sure it matches your best run (e.g., downsample_factor: 30)
    # assert config['load_rgb'] is True
    # assert config['downsample_factor'] == 30

    # 2. Load Training Data
    print("[INFO]: Loading training dataset...")
    images, distances = load_dataset(config, split="train")
    print(f"[INFO]: Training dataset loaded: {images.shape=}, {distances.shape=}")

    # 3. Data Splitting (Train/Validation)
    print("[INFO]: Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        images, distances, test_size=0.2, random_state=42 # Use fixed seed
    )
    print(f"[INFO]: Data split: {X_train.shape=}, {X_val.shape=}")

    # 4. Define the Preprocessing and Model Pipeline
    # *** UPDATED FOR RandomForestRegressor ***
    print("[INFO]: Defining the pipeline with RandomForestRegressor...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Scale features
        # Keep PCA step - n_components will be tuned by GridSearchCV
        ('pca', PCA(random_state=42)),
        # *** CHANGE REGRESSOR HERE to RandomForestRegressor ***
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1)) # Use n_jobs=-1 for speed
    ])

    # 5. Define the Hyperparameter Grid for RandomForestRegressor
    # *** NEW param_grid for RandomForestRegressor ***
    print("[INFO]: Defining parameter grid for RandomForestRegressor...")
    param_grid = {
        # Test PCA around the low values that worked for HGBR (e.g., 20) on this data
        'pca__n_components': [10, 15, 20, 30, 50],
        # Typical RandomForest parameters to tune
        'regressor__n_estimators': [100, 200, 300, 400], # Number of trees
        'regressor__max_depth': [None, 10, 20, 30], # Max depth (None means no limit)
        'regressor__min_samples_split': [2, 5, 10], # Min samples to split a node
        'regressor__min_samples_leaf': [1, 3, 5] # Min samples in a leaf node
        # 'regressor__max_features': ['sqrt', 'log2'] # Often defaults are good ('sqrt' for RF)
    }
    print(f"[INFO]: Parameter grid to search:\n{param_grid}")


    # 6. Perform Grid Search with Cross-Validation
    print("[INFO]: Setting up GridSearchCV...")
    # Use negative MAE because GridSearchCV maximizes score
    neg_mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Use the TRAINING portion for GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5, # 5-fold cross-validation
        scoring=neg_mae_scorer,
        n_jobs=-1, # Use all available CPU cores
        verbose=1 # Set verbose level (e.g., 1 for less output)
    )

    print("[INFO]: Starting Grid Search CV on the training data...")
    grid_search.fit(X_train, y_train) # Fit only on the training split

    # 7. Display Best Results from Cross-Validation
    print(f"[INFO]: Best parameters found by CV: {grid_search.best_params_}")
    # Score is negative MAE, convert to positive MAE in cm
    best_cv_mae_cm = -grid_search.best_score_ * 100
    print(f"[INFO]: Best cross-validation MAE: {best_cv_mae_cm:.3f} cm")

    # The best estimator found by GridSearch (already refit on X_train, y_train)
    best_pipeline = grid_search.best_estimator_

    # 8. Evaluate on the *Validation* Set
    print("[INFO]: Evaluating the best pipeline on the validation set...")
    val_pred = best_pipeline.predict(X_val)
    # Use the provided print_results function which calculates MAE in cm
    print_results(y_val, val_pred) # Prints MAE in cm and R2 score

    # 9. Load Test Data
    print("[INFO]: Loading test dataset...")
    test_images = load_test_dataset(config)
    # Convert list of arrays to a 2D numpy array for prediction
    test_images_np = np.array(test_images)
    print(f"[INFO]: Test dataset loaded: {test_images_np.shape=}")

    # 10. Make Predictions on Test Data using the Best Pipeline
    print("[INFO]: Making predictions on test data...")
    test_pred = best_pipeline.predict(test_images_np)

    # 11. Save Predictions for Kaggle Submission
    print("[INFO]: Saving predictions to prediction.csv...")
    save_results(test_pred)

    # 12. Save the Trained Pipeline for Moodle Submission
    # This pipeline was trained on X_train, y_train via GridSearchCV's refit
    print("[INFO]: Saving the best pipeline to pipeline.joblib...")
    joblib.dump(best_pipeline, 'pipeline.joblib')

    print("[INFO]: --- Script Finished ---")