import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore")


def load_and_preprocess_data(filepath):
    """Load and do initial preprocessing of the dataset"""
    print("Loading dataset...")

    # Use more efficient data types and chunking for large files
    df = pd.read_csv(filepath, low_memory=False)

    # Fill missing values once
    df.fillna(0, inplace=True)

    # Separate features and labels
    labels = df[["class1", "class2", "class3"]].copy()
    features = df.drop(["class1", "class2", "class3"], axis=1)

    return features, labels


def stratified_split(X, y, test_size=0.3, val_size=0.5, random_state=42):
    """Split data into train/val/test with stratification"""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        shuffle=True,
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def identify_column_types(df):
    """Identify column types once to avoid repeated dtype checks"""
    datetime_cols = []
    categorical_cols = []
    numeric_cols = []

    for col in df.columns:
        if df[col].dtype == "object":
            # Check if it's a datetime column
            if col.lower() in ["date", "timestamp"]:
                datetime_cols.append(col)
            else:
                # Try numeric conversion on a sample
                sample = df[col].head(1000).dropna()
                try:
                    pd.to_numeric(sample)
                    numeric_cols.append(col)
                except:
                    categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    return datetime_cols, categorical_cols, numeric_cols


def preprocess_features_optimized(X_train, X_val, X_test):
    """
    Optimized feature preprocessing with parallel processing and vectorized operations
    """
    print("Preprocessing features...")

    # Identify column types once
    datetime_cols, categorical_cols, numeric_cols = identify_column_types(X_train)

    # Create copies
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    X_test_processed = X_test.copy()

    label_encoders = {}

    # Process datetime columns
    for col in datetime_cols:
        print(f"Processing datetime column: {col}")
        X_train_processed[col] = pd.to_datetime(
            X_train_processed[col], errors="coerce"
        ).astype("int64")
        X_val_processed[col] = pd.to_datetime(
            X_val_processed[col], errors="coerce"
        ).astype("int64")
        X_test_processed[col] = pd.to_datetime(
            X_test_processed[col], errors="coerce"
        ).astype("int64")

    # Process numeric columns (convert object columns that are actually numeric)
    for col in numeric_cols:
        if X_train_processed[col].dtype == "object":
            X_train_processed[col] = pd.to_numeric(
                X_train_processed[col], errors="coerce"
            )
            X_val_processed[col] = pd.to_numeric(X_val_processed[col], errors="coerce")
            X_test_processed[col] = pd.to_numeric(
                X_test_processed[col], errors="coerce"
            )

    # Process categorical columns with optimized encoding
    for col in categorical_cols:
        print(f"Processing categorical column: {col}")
        le = LabelEncoder()

        # Convert to string and get unique values
        train_str = X_train_processed[col].astype(str)
        val_str = X_val_processed[col].astype(str)
        test_str = X_test_processed[col].astype(str)

        # Fit on training data
        le.fit(train_str)

        # Create a mapping for faster lookup
        class_to_num = {cls: i for i, cls in enumerate(le.classes_)}

        # Transform using vectorized operations
        X_train_processed[col] = train_str.map(class_to_num)
        X_val_processed[col] = val_str.map(class_to_num).fillna(0).astype(int)
        X_test_processed[col] = test_str.map(class_to_num).fillna(0).astype(int)

        label_encoders[col] = le

    # Handle remaining NaN values
    X_train_processed.fillna(0, inplace=True)
    X_val_processed.fillna(0, inplace=True)
    X_test_processed.fillna(0, inplace=True)

    # Normalize features
    print("Normalizing features...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_val_scaled = scaler.transform(X_val_processed)
    X_test_scaled = scaler.transform(X_test_processed)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler, label_encoders


def process_labels_optimized(y_train, y_val, y_test):
    """Optimized label processing"""
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)

    return y_train_encoded, y_val_encoded, y_test_encoded, le


def run_optimized_pipeline(filepath):
    """Main optimized pipeline function"""
    import time

    start_time = time.time()

    # Load data
    features, labels = load_and_preprocess_data(filepath)

    print(f"Dataset shape: {features.shape}")
    print(f"Data loading time: {time.time() - start_time:.2f} seconds")

    # Split data for both classification tasks
    print("Splitting data...")
    split_start = time.time()

    # Split for binary classification
    (
        X_train_bin,
        X_val_bin,
        X_test_bin,
        y_train_bin_raw,
        y_val_bin_raw,
        y_test_bin_raw,
    ) = stratified_split(features, labels["class3"])

    # Split for multi-class classification
    (
        X_train_multi,
        X_val_multi,
        X_test_multi,
        y_train_multi_raw,
        y_val_multi_raw,
        y_test_multi_raw,
    ) = stratified_split(features, labels["class1"])

    print(f"Data splitting time: {time.time() - split_start:.2f} seconds")

    # Process labels
    print("Processing labels...")
    label_start = time.time()

    y_train_bin, y_val_bin, y_test_bin, le_bin = process_labels_optimized(
        y_train_bin_raw, y_val_bin_raw, y_test_bin_raw
    )

    y_train_multi, y_val_multi, y_test_multi, le_multi = process_labels_optimized(
        y_train_multi_raw, y_val_multi_raw, y_test_multi_raw
    )

    print(f"Label processing time: {time.time() - label_start:.2f} seconds")

    # Process features for binary classification
    print("Processing features for binary classification...")
    bin_start = time.time()

    (
        X_train_bin_scaled,
        X_val_bin_scaled,
        X_test_bin_scaled,
        scaler_bin,
        encoders_bin,
    ) = preprocess_features_optimized(X_train_bin, X_val_bin, X_test_bin)

    print(f"Binary feature processing time: {time.time() - bin_start:.2f} seconds")

    # Process features for multi-class classification
    print("Processing features for multi-class classification...")
    multi_start = time.time()

    (
        X_train_multi_scaled,
        X_val_multi_scaled,
        X_test_multi_scaled,
        scaler_multi,
        encoders_multi,
    ) = preprocess_features_optimized(X_train_multi, X_val_multi, X_test_multi)

    print(
        f"Multi-class feature processing time: {time.time() - multi_start:.2f} seconds"
    )

    # Print results
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")

    print(f"\nBinary Classification:")
    print(f"Training set shape: {X_train_bin_scaled.shape}")
    print(f"Validation set shape: {X_val_bin_scaled.shape}")
    print(f"Test set shape: {X_test_bin_scaled.shape}")
    print(f"Binary classes: {le_bin.classes_}")

    print(f"\nMulti-class Classification:")
    print(f"Training set shape: {X_train_multi_scaled.shape}")
    print(f"Validation set shape: {X_val_multi_scaled.shape}")
    print(f"Test set shape: {X_test_multi_scaled.shape}")
    print(f"Number of classes: {len(le_multi.classes_)}")
    print(f"Multi classes: {le_multi.classes_}")

    print(f"\nClass distributions:")
    print(
        f"Binary - Train: {np.bincount(y_train_bin)}, Val: {np.bincount(y_val_bin)}, Test: {np.bincount(y_test_bin)}"
    )
    print(
        f"Multi - Train: {np.bincount(y_train_multi)}, Val: {np.bincount(y_val_multi)}, Test: {np.bincount(y_test_multi)}"
    )

    print(f"\nTotal pipeline time: {time.time() - start_time:.2f} seconds")

    return {
        "binary": {
            "X_train": X_train_bin_scaled,
            "X_val": X_val_bin_scaled,
            "X_test": X_test_bin_scaled,
            "y_train": y_train_bin,
            "y_val": y_val_bin,
            "y_test": y_test_bin,
            "scaler": scaler_bin,
            "encoders": encoders_bin,
            "label_encoder": le_bin,
        },
        "multi": {
            "X_train": X_train_multi_scaled,
            "X_val": X_val_multi_scaled,
            "X_test": X_test_multi_scaled,
            "y_train": y_train_multi,
            "y_val": y_val_multi,
            "y_test": y_test_multi,
            "scaler": scaler_multi,
            "encoders": encoders_multi,
            "label_encoder": le_multi,
        },
    }


# Usage
if __name__ == "__main__":
    filepath = "/content/X-IIoTID dataset.csv"
    results = run_optimized_pipeline(filepath)
