
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier
)

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =========================================================
# 1. REGRESSION MODELS
# =========================================================

def get_regression_model(name, random_state=42):

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state
        )
    }

    if name == "XGBoost":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=random_state,
                n_jobs=-1,
                objective="reg:squarederror"
            )

        except ImportError:
            raise ImportError(
                "Install XGBoost using: pip install xgboost"
            )

    if name not in models:
        raise ValueError(
            f"Unknown regression model: {name}"
        )

    return models[name]


# =========================================================
# 2. CLASSIFICATION MODELS
# =========================================================

def get_classification_model(name, random_state=42):

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1
        )
    }

    if name not in models:
        raise ValueError(
            f"Unknown classification model: {name}"
        )

    return models[name]


# =========================================================
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# =========================================================

def chronological_split(
    X,
    y,
    test_size=0.2,
    gap=1
):

    X = X.copy()
    y = y.copy()

    if not X.index.equals(y.index):
        y = y.reindex(X.index)

    if isinstance(X.index, pd.MultiIndex):

        dates = (
            X.index.get_level_values(0)
            .unique()
            .sort_values()
        )

        if len(dates) < 3:
            raise ValueError(
                "Not enough unique dates for chronological splitting."
            )

        split_index = int(
            len(dates) * (1 - test_size)
        )

        test_start_index = split_index + gap

        if (
            split_index <= 0
            or test_start_index >= len(dates)
        ):
            raise ValueError(
                "Insufficient dates for the selected test size and gap."
            )

        train_cutoff = dates[split_index]
        test_start = dates[test_start_index]

        train_mask = (
            X.index.get_level_values(0) < train_cutoff
        )

        test_mask = (
            X.index.get_level_values(0) >= test_start
        )

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]

        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]

    else:

        split_index = int(
            len(X) * (1 - test_size)
        )

        test_start_index = split_index + gap

        if (
            split_index <= 0
            or test_start_index >= len(X)
        ):
            raise ValueError(
                "Invalid chronological split."
            )

        X_train = X.iloc[:split_index]
        X_test = X.iloc[test_start_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[test_start_index:]

    if X_train.empty or X_test.empty:
        raise ValueError(
            "Training or testing dataset is empty."
        )

    return X_train, X_test, y_train, y_test


# =========================================================
# 4. DATA PREPARATION
# =========================================================

def _prepare_model_data(X, y):

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a Pandas DataFrame.")

    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X.index)

    if not X.index.equals(y.index):
        y = y.reindex(X.index)

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    valid_target = y.notna()

    X = X.loc[valid_target]
    y = y.loc[valid_target]

    X = X.dropna(
        axis=1,
        how="all"
    )

    if X.empty:
        raise ValueError(
            "No usable model features available."
        )

    if len(X) < 10:
        raise ValueError(
            "Insufficient observations for model training."
        )

    return X, y


# =========================================================
# 5. REGRESSION TRAINING
# =========================================================

def train_regression_model(
    X,
    y,
    model_name="Random Forest",
    test_size=0.2,
    random_state=42,
    gap=1
):

    X, y = _prepare_model_data(X, y)

    X_train, X_test, y_train, y_test = chronological_split(
        X,
        y,
        test_size=test_size,
        gap=gap
    )

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        ),

        (
            "model",
            get_regression_model(
                model_name,
                random_state
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    metrics = {
        "R Squared": r2_score(
            y_test,
            predictions
        ),

        "MAE": mean_absolute_error(
            y_test,
            predictions
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )
    }

    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": predictions
    }, index=y_test.index)

    return model, metrics, results


# =========================================================
# 6. CLASSIFICATION TRAINING
# =========================================================

def train_classification_model(
    X,
    y,
    model_name="Logistic Regression",
    test_size=0.2,
    random_state=42,
    gap=1
):

    X, y = _prepare_model_data(X, y)

    y = y.astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "Classification requires at least two target classes."
        )

    X_train, X_test, y_train, y_test = chronological_split(
        X,
        y,
        test_size=test_size,
        gap=gap
    )

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data contains only one class."
        )

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            get_classification_model(
                model_name,
                random_state
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    }

    if y_test.nunique() == 2:

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        metrics["ROC AUC"] = roc_auc_score(
            y_test,
            probabilities
        )

    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": predictions
    }, index=y_test.index)

    if hasattr(model, "predict_proba"):

        results["Positive_Probability"] = (
            model.predict_proba(X_test)[:, 1]
        )

    return model, metrics, results
