from src.model_training import (
    get_train_test_data,
    build_preprocessor,
    build_models,
)


def test_train_test_split_stratified():
    X_train, X_test, y_train, y_test = get_train_test_data()
    # Non-empty splits
    assert len(X_train) > 0
    assert len(X_test) > 0
    # Target values only 0/1
    assert set(y_train.unique()) <= {0, 1}
    assert set(y_test.unique()) <= {0, 1}


def test_build_models_pipelines():
    X_train, X_test, y_train, y_test = get_train_test_data()
    preprocessor = build_preprocessor()
    lr_pipeline, rf_pipeline = build_models(preprocessor)

    assert hasattr(lr_pipeline, "fit")
    assert hasattr(lr_pipeline, "predict")
    assert hasattr(rf_pipeline, "fit")
    assert hasattr(rf_pipeline, "predict")

    lr_pipeline.fit(X_train.head(50), y_train.head(50))
    rf_pipeline.fit(X_train.head(50), y_train.head(50))
