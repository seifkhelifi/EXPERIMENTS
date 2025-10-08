import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


def load_and_preprocess_data(filepath):
    """Load and do initial preprocessing of the dataset"""
    print("Loading dataset...")

    # Use more efficient data types and chunking for large files
    df = pd.read_csv(filepath, low_memory=False)

    # Fill missing values once
    df.fillna(0, inplace=True)

    # Separate features and labels - only need class1 for multi-class
    labels = df["class1"].copy()
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


def apply_smote_with_visualization(
    X_train,
    y_train,
    label_encoder=None,
    task_name="Multi-class Classification",
    sampling_strategy="auto",
    random_state=42,
    k_neighbors=5,
):

    print(f"\nApplying SMOTE for {task_name}...")

    # Get original class names if label encoder is provided
    if label_encoder is not None:
        class_names = label_encoder.classes_
    else:
        class_names = [f"Class {i}" for i in np.unique(y_train)]

    # Count original class distribution
    original_counts = np.bincount(y_train)
    print(f"Original class distribution: {original_counts}")

    # Apply SMOTE
    try:
        # Handle edge case where k_neighbors might be too large
        n_samples_min_class = np.min(original_counts[original_counts > 0])
        if k_neighbors >= n_samples_min_class:
            k_neighbors = max(1, n_samples_min_class - 1)
            print(f"Adjusted k_neighbors to {k_neighbors} due to small class size")

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )

        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # Count new class distribution
        new_counts = np.bincount(y_resampled)
        print(f"New class distribution: {new_counts}")
        print(f"Training samples increased from {len(y_train)} to {len(y_resampled)}")

    except Exception as e:
        print(f"SMOTE failed with error: {e}")
        print("Returning original data without SMOTE...")
        return X_train, y_train

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot original distribution
    bars1 = ax1.bar(
        range(len(original_counts)),
        original_counts,
        color="skyblue",
        alpha=0.7,
        edgecolor="black",
    )
    ax1.set_xlabel("Classes")
    ax1.set_ylabel("Number of Samples")
    ax1.set_title(f"{task_name} - Original Class Distribution")
    ax1.set_xticks(range(len(original_counts)))
    ax1.set_xticklabels(
        [
            class_names[i] if i < len(class_names) else f"Class {i}"
            for i in range(len(original_counts))
        ],
        rotation=45,
    )

    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    # Plot new distribution
    bars2 = ax2.bar(
        range(len(new_counts)),
        new_counts,
        color="lightcoral",
        alpha=0.7,
        edgecolor="black",
    )
    ax2.set_xlabel("Classes")
    ax2.set_ylabel("Number of Samples")
    ax2.set_title(f"{task_name} - After SMOTE Class Distribution")
    ax2.set_xticks(range(len(new_counts)))
    ax2.set_xticklabels(
        [
            class_names[i] if i < len(class_names) else f"Class {i}"
            for i in range(len(new_counts))
        ],
        rotation=45,
    )

    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\n{task_name} SMOTE Summary:")
    print(f"{'Class':<15} {'Original':<10} {'After SMOTE':<12} {'Increase':<10}")
    print("-" * 50)
    for i in range(len(original_counts)):
        class_name = class_names[i] if i < len(class_names) else f"Class {i}"
        original = original_counts[i] if i < len(original_counts) else 0
        new = new_counts[i] if i < len(new_counts) else 0
        increase = new - original
        print(f"{class_name:<15} {original:<10} {new:<12} {increase:<10}")

    return X_resampled, y_resampled


def run_optimized_pipeline_with_smote(
    filepath, apply_smote=False, smote_strategy="auto"
):
    """Main optimized pipeline function for multi-class classification only"""
    import time

    print("\n" + "=" * 80)
    print("Running pipeline with SMOTE for multi-class classification...")
    print("=" * 80)

    start_time = time.time()

    # Load data
    features, labels = load_and_preprocess_data(filepath)

    print(f"Dataset shape: {features.shape}")
    print(f"Data loading time: {time.time() - start_time:.2f} seconds")

    # Split data for multi-class classification only
    print("Splitting data...")
    split_start = time.time()

    X_train, X_val, X_test, y_train_raw, y_val_raw, y_test_raw = stratified_split(
        features, labels
    )

    print(f"Data splitting time: {time.time() - split_start:.2f} seconds")

    # Process labels
    print("Processing labels...")
    label_start = time.time()

    y_train, y_val, y_test, label_encoder = process_labels_optimized(
        y_train_raw, y_val_raw, y_test_raw
    )

    print(f"Label processing time: {time.time() - label_start:.2f} seconds")

    # Process features
    print("Processing features...")
    feature_start = time.time()

    X_train_scaled, X_val_scaled, X_test_scaled, scaler, encoders = (
        preprocess_features_optimized(X_train, X_val, X_test)
    )

    print(f"Feature processing time: {time.time() - feature_start:.2f} seconds")

    # Apply SMOTE if requested
    if apply_smote:
        print("\n" + "=" * 60)
        print("APPLYING SMOTE TO MULTI-CLASS TRAINING DATA")
        print("=" * 60)

        smote_start = time.time()

        X_train_smote, y_train_smote = apply_smote_with_visualization(
            X_train_scaled,
            y_train,
            label_encoder=label_encoder,
            task_name="Multi-class Classification",
            sampling_strategy=smote_strategy,
        )

        print(f"SMOTE processing time: {time.time() - smote_start:.2f} seconds")

        # Update training data with SMOTE results
        X_train_final = X_train_smote
        y_train_final = y_train_smote
    else:
        X_train_final = X_train_scaled
        y_train_final = y_train

    # Print results
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS - MULTI-CLASS CLASSIFICATION")
    print(f"{'='*50}")

    print(f"Training set shape: {X_train_final.shape}")
    print(f"Validation set shape: {X_val_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")

    print(f"\nClass distributions:")
    print(f"Train: {np.bincount(y_train_final)}")
    print(f"Val: {np.bincount(y_val)}")
    print(f"Test: {np.bincount(y_test)}")

    print(f"\nTotal pipeline time: {time.time() - start_time:.2f} seconds")

    return {
        "multi": {
            "X_train": X_train_final,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train_final,
            "y_val": y_val,
            "y_test": y_test,
            "scaler": scaler,
            "encoders": encoders,
            "label_encoder": label_encoder,
        }
    }


# Usage examples
if __name__ == "__main__":
    filepath = "./X-IIoTID dataset.csv"

    # Run without SMOTE
    print("Running pipeline without SMOTE...")

    # Run with SMOTE for multi-class classification

    results_with_smote = run_optimized_pipeline_with_smote(
        filepath, apply_smote=True, smote_strategy="auto"
    )
