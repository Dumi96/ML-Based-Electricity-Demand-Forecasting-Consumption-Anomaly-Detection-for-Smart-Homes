from xgboost import XGBClassifier


def create_weighted_peak_classifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=19.0,
    random_state=42,
    n_jobs=-1
):
    """
    Create a class-weighted XGBoost classifier
    for electricity peak-demand classification.
    """

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs
    )

    return model