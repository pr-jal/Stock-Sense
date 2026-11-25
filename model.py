import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES

SEED = 42


def make_models() -> dict:
    return {
        "Baseline (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=20,
            random_state=SEED, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200,
            min_samples_leaf=30, random_state=SEED),
    }


def chrono_split(data: pd.DataFrame, train=0.70, val=0.15):
    n = len(data)
    i_train, i_val = int(n * train), int(n * (train + val))
    return data.iloc[:i_train], data.iloc[i_train:i_val], data.iloc[i_val:]


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_pipeline(data: pd.DataFrame, chosen: str = None) -> dict:
    if len(data) < 300:
        raise ValueError(f"There are only {len(data)} usable trading days. Please choose a longer "
                         f"date range (at least 2 years, ideally 5 or more).")

    train, val, test = chrono_split(data)
    X_tr, y_tr = train[FEATURES], train["target"].astype(int)
    X_va, y_va = val[FEATURES], val["target"].astype(int)
    X_te, y_te = test[FEATURES], test["target"].astype(int)

    val_rows = []
    for name, model in make_models().items():
        model.fit(X_tr, y_tr)
        val_rows.append({"model": name, **metrics(y_va, model.predict(X_va))})
    val_table = pd.DataFrame(val_rows).set_index("model")

    baseline_name = "Baseline (majority class)"
    ml_only = val_table.drop(index=baseline_name)
    best_name = ml_only["accuracy"].idxmax()
    beats_baseline_val = ml_only.loc[best_name, "accuracy"] > val_table.loc[baseline_name, "accuracy"]
    auto_choice = best_name
    if chosen:
        best_name = chosen

    X_trva = pd.concat([X_tr, X_va])
    y_trva = pd.concat([y_tr, y_va])
    test_rows, test_preds, test_probs, fitted = [], {}, {}, {}
    for name, model in make_models().items():
        model.fit(X_trva, y_trva)
        pred = model.predict(X_te)
        test_rows.append({"model": name, **metrics(y_te, pred)})
        test_preds[name] = pred
        test_probs[name] = model.predict_proba(X_te)[:, 1]
        fitted[name] = model
    test_table = pd.DataFrame(test_rows).set_index("model")

    return {
        "periods": {
            "train": (train.index.min(), train.index.max(), len(train)),
            "validation": (val.index.min(), val.index.max(), len(val)),
            "test": (test.index.min(), test.index.max(), len(test)),
        },
        "val_table": val_table,
        "test_table": test_table,
        "selected_model": best_name,
        "auto_choice": auto_choice,
        "beats_baseline_on_validation": bool(beats_baseline_val),
        "confusion_matrix": confusion_matrix(y_te, test_preds[best_name], labels=[0, 1]),
        "test_df": test,
        "test_prob": pd.Series(test_probs[best_name], index=test.index),
        "test_pred": pd.Series(test_preds[best_name], index=test.index),
        "up_share_test": float(y_te.mean()),
        "feature_importance": feature_importance(fitted[best_name], X_te, y_te),
    }


def feature_importance(model, X, y) -> pd.Series:
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, n_repeats=10, random_state=SEED, scoring="accuracy")
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def backtest(test_df: pd.DataFrame, prob: pd.Series, threshold=0.5, cost=0.001) -> pd.DataFrame:
    next_ret = test_df["Close"].pct_change().shift(-1)
    position = (prob > threshold).astype(int)
    trades = position.diff().abs().fillna(position.iloc[0])
    strat_ret = position * next_ret - trades * cost
    out = pd.DataFrame({
        "Model strategy": (1 + strat_ret.fillna(0)).cumprod(),
        "Buy and hold": (1 + next_ret.fillna(0)).cumprod(),
    })
    return out


def predict_next_day(data: pd.DataFrame, latest: pd.DataFrame, model_name: str):
    model = make_models()[model_name]
    model.fit(data[FEATURES], data["target"].astype(int))
    p_up = float(model.predict_proba(latest[FEATURES])[0, 1])
    return p_up, model


def explain_prediction(model, data: pd.DataFrame, latest: pd.DataFrame) -> pd.Series:
    base = float(model.predict_proba(latest[FEATURES])[0, 1])
    typical = data[FEATURES].median()
    effects = {}
    for f in FEATURES:
        what_if = latest[FEATURES].copy()
        what_if[f] = typical[f]
        effects[f] = base - float(model.predict_proba(what_if)[0, 1])
    return pd.Series(effects).sort_values(key=abs, ascending=False)


def walk_forward(data: pd.DataFrame, model_name: str, n_folds: int = 5) -> pd.DataFrame:
    from sklearn.model_selection import TimeSeriesSplit
    X, y = data[FEATURES], data["target"].astype(int)
    rows = []
    for i, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_folds).split(X), start=1):
        model = make_models()[model_name].fit(X.iloc[tr], y.iloc[tr])
        base = make_models()["Baseline (majority class)"].fit(X.iloc[tr], y.iloc[tr])
        rows.append({
            "Fold": i,
            "Train from": X.index[tr[0]].date(), "Train to": X.index[tr[-1]].date(),
            "Test from": X.index[te[0]].date(), "Test to": X.index[te[-1]].date(),
            "Model accuracy": accuracy_score(y.iloc[te], model.predict(X.iloc[te])),
            "Baseline accuracy": accuracy_score(y.iloc[te], base.predict(X.iloc[te])),
        })
    return pd.DataFrame(rows).set_index("Fold")
