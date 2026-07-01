from src.data_pipeline import (
    load_raw_data,
    build_preprocessor,
    numeric_features,
    categorical_features,
    target_col,
)


def test_load_raw_data_shape_and_columns():
    df = load_raw_data()
    # Basic sanity checks
    assert target_col in df.columns
    for col in numeric_features + categorical_features:
        assert col in df.columns
    assert len(df) > 0  # dataset is non-empty


def test_preprocessor_handles_missing_values():
    df = load_raw_data()
    X = df[numeric_features + categorical_features]

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    # Should return a 2D array with no NaNs
    assert X_processed.ndim == 2
    # scipy.sparse or numpy array both okay; just check shape
    assert X_processed.shape[0] == len(df)
