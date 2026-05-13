"""ML Predictive Model - Supervised learning pipeline with Scikit-learn."""
  import numpy as np
  import pandas as pd
  from sklearn.datasets import make_classification
  from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
  from sklearn.preprocessing import StandardScaler, LabelEncoder
  from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.pipeline import Pipeline
  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
  from sklearn.impute import SimpleImputer
  import joblib, warnings
  warnings.filterwarnings("ignore")

  # ── Data ─────────────────────────────────────────────────────────────────
  X, y = make_classification(n_samples=5000, n_features=20, n_informative=15, random_state=42)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

  # ── Preprocessing Pipeline ────────────────────────────────────────────────
  preprocessor = Pipeline([
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler())
  ])

  X_train = preprocessor.fit_transform(X_train)
  X_test = preprocessor.transform(X_test)

  # ── Model Comparison ──────────────────────────────────────────────────────
  models = {
      "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
      "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
      "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
  }

  print("\n── Model Comparison (5-fold CV) ──────────────────────────")
  best_model, best_score = None, 0
  for name, model in models.items():
      cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
      print(f"{name:25s}  CV Acc: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
      if cv_scores.mean() > best_score:
          best_score, best_model = cv_scores.mean(), (name, model)

  # ── Hyperparameter Tuning ─────────────────────────────────────────────────
  print(f"\n── Tuning best model: {best_model[0]} ──────────────────────")
  param_grid = {"n_estimators": [100, 200], "max_depth": [None, 10, 20]} if "Forest" in best_model[0] else {}
  if param_grid:
      gs = GridSearchCV(best_model[1], param_grid, cv=3, scoring="accuracy", n_jobs=-1)
      gs.fit(X_train, y_train)
      final_model = gs.best_estimator_
      print(f"Best params: {gs.best_params_}")
  else:
      final_model = best_model[1]
      final_model.fit(X_train, y_train)

  # ── Evaluation ────────────────────────────────────────────────────────────
  y_pred = final_model.predict(X_test)
  print(f"\n── Test Set Evaluation ─────────────────────────────────────")
  print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
  print(f"Precision: {precision_score(y_test, y_pred):.4f}")
  print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
  print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
  print("\n" + classification_report(y_test, y_pred))

  # ── Save ─────────────────────────────────────────────────────────────────
  joblib.dump(final_model, "model.joblib")
  joblib.dump(preprocessor, "preprocessor.joblib")
  print("\nModel saved: model.joblib | preprocessor.joblib")
  