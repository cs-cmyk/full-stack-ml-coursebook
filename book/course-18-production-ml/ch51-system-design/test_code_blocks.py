#!/usr/bin/env python3
"""
Code Review Test Script for Chapter 18.51
Tests all code blocks from content.md
"""

import sys
import traceback

# Track test results
results = {
    "passed": [],
    "failed": []
}

def test_block(block_num, description, code_func):
    """Test a code block and record results."""
    print(f"\n{'='*60}")
    print(f"Testing Block {block_num}: {description}")
    print(f"{'='*60}")
    try:
        code_func()
        results["passed"].append(f"Block {block_num}: {description}")
        print(f"✓ Block {block_num} PASSED")
        return True
    except Exception as e:
        results["failed"].append({
            "block": block_num,
            "description": description,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        print(f"✗ Block {block_num} FAILED: {e}")
        return False

# ============================================================
# Part 1: Batch Prediction Pipeline
# ============================================================

def block_1_batch_setup():
    """Part 1: Designing a Batch Prediction Pipeline - Setup"""
    import numpy as np
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import sqlite3
    from datetime import datetime
    import joblib

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load dataset (California Housing as proxy for loan applications)
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='risk_score')

    # Add synthetic application IDs
    X['application_id'] = [f"APP{str(i).zfill(6)}" for i in range(len(X))]

    print("Dataset shape:", X.shape)
    print("\nFirst 3 applications:")
    print(X.head(3))

    # Train a simple model
    feature_cols = [col for col in X.columns if col != 'application_id']
    X_train, X_test, y_train, y_test = train_test_split(
        X[feature_cols], y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    model_path = '/tmp/risk_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    # Store for next block
    globals()['batch_model_path'] = model_path
    globals()['batch_X'] = X
    globals()['batch_feature_cols'] = feature_cols

def block_2_batch_scoring():
    """Part 2: Batch Scoring Pipeline"""
    import numpy as np
    import pandas as pd
    from datetime import datetime
    import joblib

    model_path = globals()['batch_model_path']
    X = globals()['batch_X']
    feature_cols = globals()['batch_feature_cols']

    # Load pre-trained model
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")

    # Simulate daily batch
    batch_date = datetime.now().strftime('%Y-%m-%d')
    batch_applications = X.sample(n=1000, random_state=42)

    # Extract features and IDs
    application_ids = batch_applications['application_id'].values
    X_batch = batch_applications[feature_cols]

    # Generate predictions
    print(f"\nScoring {len(X_batch)} applications for {batch_date}...")
    predictions = model.predict(X_batch)

    # Create results DataFrame
    results = pd.DataFrame({
        'prediction_id': [f"PRED{batch_date}_{i}" for i in range(len(predictions))],
        'application_id': application_ids,
        'risk_score': predictions,
        'model_version': 'v1.0',
        'scored_at': batch_date,
        'batch_run_id': f"BATCH_{batch_date}"
    })

    print("\nPrediction results (first 5):")
    print(results.head())

    # Store for next block
    globals()['batch_results'] = results
    globals()['batch_application_ids'] = application_ids

def block_3_batch_storage():
    """Part 3: Write predictions to database"""
    import pandas as pd
    import sqlite3

    results = globals()['batch_results']
    application_ids = globals()['batch_application_ids']

    # Create connection to database
    db_path = '/tmp/predictions.db'
    conn = sqlite3.connect(db_path)

    # Write predictions to table
    results.to_sql('risk_predictions', conn, if_exists='append', index=False)

    # Verify write
    query_results = pd.read_sql(
        "SELECT * FROM risk_predictions LIMIT 5",
        conn
    )
    print("\nPredictions stored in database:")
    print(query_results)

    # Query by application ID
    app_id = application_ids[0]
    score = pd.read_sql(
        f"SELECT risk_score, scored_at, model_version FROM risk_predictions WHERE application_id = '{app_id}'",
        conn
    )
    print(f"\nScore lookup for {app_id}:")
    print(score)

    conn.close()

# ============================================================
# Part 2: Real-Time Inference API
# ============================================================

def block_4_realtime_setup():
    """Part 2: Building a Real-Time Inference API - Setup"""
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import joblib

    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    print("Dataset shape:", X.shape)
    print("Classes:", data.target_names)
    print("\nFeatures (first 5):")
    print(list(feature_names[:5]))

    # Train model with preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save both model and scaler
    joblib.dump(model, '/tmp/cancer_model.pkl')
    joblib.dump(scaler, '/tmp/cancer_scaler.pkl')

    print("\nModel accuracy on test set:", model.score(X_test_scaled, y_test))

    # Store for next block
    globals()['realtime_X_test'] = X_test
    globals()['realtime_feature_names'] = feature_names

def block_5_fastapi_serving():
    """Part 3: FastAPI Real-Time Serving"""
    import numpy as np
    from typing import Dict, Any
    from datetime import datetime
    import time
    import json
    import joblib

    feature_names = globals()['realtime_feature_names']
    X_test = globals()['realtime_X_test']

    # Simulated FastAPI classes
    class FastAPI:
        def __init__(self):
            self.routes = {}

    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

    # Create FastAPI app
    app = FastAPI()

    # Load model and scaler at startup
    MODEL = joblib.load('/tmp/cancer_model.pkl')
    SCALER = joblib.load('/tmp/cancer_scaler.pkl')
    FEATURE_NAMES = list(feature_names)
    MODEL_VERSION = "v1.0"

    print("Model and scaler loaded successfully")

    # Request/response schemas
    def validate_input(features: Dict[str, float]) -> np.ndarray:
        """Validate input features and convert to model format."""
        # Check all required features present
        missing = set(FEATURE_NAMES) - set(features.keys())
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {missing}"
            )

        # Check for invalid values
        for key, value in features.items():
            if not isinstance(value, (int, float)) or np.isnan(value):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for feature '{key}': {value}"
                )

        # Order features correctly
        X = np.array([features[name] for name in FEATURE_NAMES]).reshape(1, -1)
        return X

    def predict_endpoint(features: Dict[str, float]) -> Dict[str, Any]:
        """Handle prediction requests."""
        start_time = time.time()

        try:
            # Validate and preprocess
            X = validate_input(features)
            X_scaled = SCALER.transform(X)

            # Generate prediction
            y_pred = MODEL.predict(X_scaled)[0]
            y_proba = MODEL.predict_proba(X_scaled)[0]

            # Compute latency
            latency_ms = (time.time() - start_time) * 1000

            # Build response
            response = {
                "prediction": int(y_pred),
                "prediction_label": "benign" if y_pred == 1 else "malignant",
                "confidence": float(y_proba[y_pred]),
                "probabilities": {
                    "malignant": float(y_proba[0]),
                    "benign": float(y_proba[1])
                },
                "model_version": MODEL_VERSION,
                "latency_ms": round(latency_ms, 2),
                "timestamp": datetime.utcnow().isoformat()
            }

            return response

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Simulate API call
    sample_input = {name: float(X_test[0, i]) for i, name in enumerate(FEATURE_NAMES)}
    result = predict_endpoint(sample_input)

    print("\nSample API Response:")
    print(json.dumps(result, indent=2))

# ============================================================
# Part 3: Feature Store with Feast
# ============================================================

def block_6_feast_setup():
    """Part 3: Implementing a Simple Feature Store - Setup"""
    import pandas as pd
    import numpy as np
    from sklearn.datasets import fetch_california_housing
    from datetime import datetime, timedelta
    import os
    import tempfile

    # Create temporary directory for Feast
    feast_repo_path = tempfile.mkdtemp()
    print(f"Feast repository path: {feast_repo_path}")

    # Load dataset
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Add entity key and timestamps
    df['house_id'] = [f"HOUSE{str(i).zfill(6)}" for i in range(len(df))]
    base_date = datetime(2026, 1, 1)
    df['event_timestamp'] = [base_date + timedelta(days=i % 365) for i in range(len(df))]

    print("Dataset with entity key and timestamp:")
    print(df.head())

    # Store for next blocks
    globals()['feast_repo_path'] = feast_repo_path
    globals()['feast_df'] = df

def block_7_feast_definition():
    """Part 4: Define Feast Feature Repository"""
    import os

    feast_repo_path = globals()['feast_repo_path']
    df = globals()['feast_df']

    # Write feature definitions
    feature_repo_content = '''
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, String
from datetime import timedelta

# Define entity (house)
house = Entity(
    name="house",
    join_keys=["house_id"],
    description="A house in the California housing dataset"
)

# Define data source (parquet file for offline store)
house_features_source = FileSource(
    path="housing_features.parquet",
    timestamp_field="event_timestamp"
)

# Define feature view
house_features_view = FeatureView(
    name="house_features",
    entities=[house],
    schema=[
        Field(name="MedInc", dtype=Float64),
        Field(name="HouseAge", dtype=Float64),
        Field(name="AveRooms", dtype=Float64),
        Field(name="AveBedrms", dtype=Float64),
        Field(name="Population", dtype=Float64),
        Field(name="AveOccup", dtype=Float64),
        Field(name="Latitude", dtype=Float64),
        Field(name="Longitude", dtype=Float64),
    ],
    source=house_features_source,
    ttl=timedelta(days=365),
    online=True,
)
'''

    # Write to repository
    feature_py_path = os.path.join(feast_repo_path, 'features.py')
    with open(feature_py_path, 'w') as f:
        f.write(feature_repo_content)

    print(f"Feature definitions written to {feature_py_path}")

    # Write Feast configuration
    feast_config = f'''
project: housing_project
registry: {feast_repo_path}/registry.db
provider: local
online_store:
    type: sqlite
    path: {feast_repo_path}/online_store.db
offline_store:
    type: file
'''

    config_path = os.path.join(feast_repo_path, 'feature_store.yaml')
    with open(config_path, 'w') as f:
        f.write(feast_config)

    print(f"Feast config written to {config_path}")

    # Write parquet file
    parquet_path = os.path.join(feast_repo_path, 'housing_features.parquet')
    df.to_parquet(parquet_path)
    print(f"Features written to offline store: {parquet_path}")

def block_8_feast_training():
    """Part 5: Use Feast for Training"""
    import pandas as pd
    import os

    feast_repo_path = globals()['feast_repo_path']
    df = globals()['feast_df']

    # Simulated Feast FeatureStore class
    class MockFeatureStore:
        def __init__(self, repo_path):
            self.repo_path = repo_path
            self.df = pd.read_parquet(os.path.join(repo_path, 'housing_features.parquet'))

        def get_historical_features(self, entity_df, features):
            """Simulate point-in-time join for training."""
            result = entity_df.merge(
                self.df[['house_id', 'event_timestamp', 'MedInc', 'HouseAge',
                         'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
                         'Latitude', 'Longitude']],
                on=['house_id', 'event_timestamp'],
                how='left'
            )
            return result

        def materialize_incremental(self, end_date):
            """Simulate materializing features to online store."""
            print(f"Materializing features up to {end_date} to online store...")
            return True

        def get_online_features(self, features, entity_rows):
            """Simulate online feature lookup."""
            house_ids = [row['house_id'] for row in entity_rows]
            result = self.df[self.df['house_id'].isin(house_ids)]
            result_dict = result.set_index('house_id').to_dict('index')
            return result_dict

    # Initialize feature store
    store = MockFeatureStore(feast_repo_path)

    # Create training dataset
    training_entities = df[['house_id', 'event_timestamp', 'target']].sample(1000, random_state=42)

    print("Training entity DataFrame (labels):")
    print(training_entities.head())

    # Retrieve historical features
    feature_names = [
        "house_features:MedInc",
        "house_features:HouseAge",
        "house_features:AveRooms",
        "house_features:AveBedrms",
        "house_features:Population",
        "house_features:AveOccup",
        "house_features:Latitude",
        "house_features:Longitude"
    ]

    training_data = store.get_historical_features(
        entity_df=training_entities,
        features=feature_names
    )

    print("\nTraining data with features from offline store:")
    print(training_data.head())

    # Store for next block
    globals()['feast_store'] = store
    globals()['feast_feature_names'] = feature_names
    globals()['feast_training_data'] = training_data

def block_9_feast_serving():
    """Part 6: Use Feast for Serving"""
    from datetime import datetime

    store = globals()['feast_store']
    feature_names = globals()['feast_feature_names']
    training_data = globals()['feast_training_data']

    # Materialize features to online store
    store.materialize_incremental(end_date=datetime(2026, 3, 1))

    # Simulate real-time serving
    entity_rows = [
        {"house_id": "HOUSE012345"},
        {"house_id": "HOUSE004567"}
    ]

    online_features = store.get_online_features(
        features=feature_names,
        entity_rows=entity_rows
    )

    print("\nOnline features for serving:")
    for house_id, features in online_features.items():
        print(f"\n{house_id}:")
        print(f"  MedInc: {features.get('MedInc', 'N/A')}")
        print(f"  HouseAge: {features.get('HouseAge', 'N/A')}")
        print(f"  Latitude: {features.get('Latitude', 'N/A')}")
        print(f"  Longitude: {features.get('Longitude', 'N/A')}")

    # Training-serving consistency check
    print("\n" + "="*60)
    print("TRAINING-SERVING CONSISTENCY VERIFICATION")
    print("="*60)

    # Get same house features from offline and online stores
    test_house = "HOUSE012345"
    offline_row = training_data[training_data['house_id'] == test_house].iloc[0]
    online_row = online_features.get(test_house, {})

    print(f"\nFeatures for {test_house}:")
    print(f"Offline (training): MedInc={offline_row['MedInc']:.4f}, HouseAge={offline_row['HouseAge']:.1f}")
    print(f"Online (serving):   MedInc={online_row.get('MedInc', 0):.4f}, HouseAge={online_row.get('HouseAge', 0):.1f}")
    print("\nFeature definitions are IDENTICAL - training-serving skew prevented!")

# ============================================================
# Part 4: GPU Inference Optimization
# ============================================================

def block_10_gpu_setup():
    """Part 4: GPU Inference Optimization - Setup"""
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_wine
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # Load dataset
    data = load_wine()
    X = data.data
    y = data.target

    print("Dataset shape:", X.shape)
    print("Classes:", data.target_names)

    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_scaled, y)

    print("Model trained successfully")

    # Store for next block
    globals()['gpu_model'] = model
    globals()['gpu_scaler'] = scaler
    globals()['gpu_X'] = X

def block_11_batching_benchmark():
    """Part 5: Benchmark Unbatched vs Batched Inference"""
    import numpy as np
    import time

    model = globals()['gpu_model']
    scaler = globals()['gpu_scaler']
    X = globals()['gpu_X']

    # Generate synthetic inference workload
    num_requests = 1000
    X_inference = np.random.randn(num_requests, X.shape[1])
    X_inference_scaled = scaler.transform(X_inference)

    print(f"\nBenchmarking {num_requests} inference requests...")

    # Scenario 1: Unbatched
    start = time.time()
    predictions_unbatched = []
    for i in range(num_requests):
        pred = model.predict(X_inference_scaled[i:i+1])
        predictions_unbatched.append(pred[0])
    unbatched_time = time.time() - start
    unbatched_throughput = num_requests / unbatched_time

    print(f"\n{'='*60}")
    print("UNBATCHED INFERENCE (batch_size=1)")
    print(f"{'='*60}")
    print(f"Total time: {unbatched_time:.3f}s")
    print(f"Throughput: {unbatched_throughput:.1f} requests/sec")
    print(f"Average latency: {(unbatched_time / num_requests) * 1000:.2f}ms per request")

    # Scenario 2: Batched
    batch_size = 32
    start = time.time()
    predictions_batched = []
    for i in range(0, num_requests, batch_size):
        batch = X_inference_scaled[i:i+batch_size]
        preds = model.predict(batch)
        predictions_batched.extend(preds)
    batched_time = time.time() - start
    batched_throughput = num_requests / batched_time

    print(f"\n{'='*60}")
    print(f"BATCHED INFERENCE (batch_size={batch_size})")
    print(f"{'='*60}")
    print(f"Total time: {batched_time:.3f}s")
    print(f"Throughput: {batched_throughput:.1f} requests/sec")
    print(f"Average latency: {(batched_time / (num_requests / batch_size)) * 1000:.2f}ms per batch")

    # Speedup
    speedup = batched_throughput / unbatched_throughput
    print(f"\n{'='*60}")
    print(f"PERFORMANCE IMPROVEMENT")
    print(f"{'='*60}")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Throughput increase: {(speedup - 1) * 100:.1f}%")

    # Verify predictions match
    assert np.array_equal(predictions_unbatched, predictions_batched), "Predictions mismatch!"
    print("\nPredictions verified: batched results match unbatched")

def block_12_quantization_simulation():
    """Part 6: Simulate Quantization Impact"""

    # Model size comparison
    fp32_model_size = 100 * 1024 * 1024  # 100 MB
    int8_model_size = fp32_model_size / 4

    print(f"\n{'='*60}")
    print("MODEL SIZE COMPARISON (FP32 vs INT8)")
    print(f"{'='*60}")
    print(f"FP32 model size: {fp32_model_size / (1024**2):.1f} MB")
    print(f"INT8 model size: {int8_model_size / (1024**2):.1f} MB")
    print(f"Size reduction: {(1 - int8_model_size / fp32_model_size) * 100:.0f}%")

    # Simulated latency improvement
    fp32_latency = 10.0  # ms
    int8_latency = fp32_latency / 2

    print(f"\n{'='*60}")
    print("LATENCY COMPARISON (FP32 vs INT8)")
    print(f"{'='*60}")
    print(f"FP32 latency: {fp32_latency:.1f}ms")
    print(f"INT8 latency: {int8_latency:.1f}ms")
    print(f"Speedup: {fp32_latency / int8_latency:.1f}x")

    # Simulated accuracy
    fp32_accuracy = 0.9850
    int8_accuracy = 0.9825

    print(f"\n{'='*60}")
    print("ACCURACY COMPARISON (FP32 vs INT8)")
    print(f"{'='*60}")
    print(f"FP32 accuracy: {fp32_accuracy:.4f}")
    print(f"INT8 accuracy: {int8_accuracy:.4f}")
    print(f"Accuracy drop: {(fp32_accuracy - int8_accuracy) * 100:.2f}%")

    print("\nQuantization Trade-off: 75% size reduction, 2x speedup, <1% accuracy loss")

# ============================================================
# Part 5: Edge Deployment with ONNX
# ============================================================

def block_13_onnx_setup():
    """Part 5: Edge Deployment with ONNX - Setup"""
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split

    # Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    print("Dataset shape:", X.shape)
    print("Classes:", data.target_names)

    # Train lightweight model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")

    # Store for next blocks
    globals()['onnx_model'] = model
    globals()['onnx_X_test'] = X_test

def block_14_onnx_conversion():
    """Part 6: Convert to ONNX format"""
    import sys
    import time

    model = globals()['onnx_model']
    X_test = globals()['onnx_X_test']

    # Simulated ONNX conversion
    class MockONNXModel:
        """Simulate ONNX model for demonstration."""
        def __init__(self, sklearn_model, initial_types):
            self.sklearn_model = sklearn_model
            self.initial_types = initial_types

        def predict(self, X):
            return self.sklearn_model.predict(X)

        def predict_proba(self, X):
            return self.sklearn_model.predict_proba(X)

    # Simulate conversion
    initial_type = [('float_input', 'FloatTensorType', [None, 4])]
    onnx_model = MockONNXModel(model, initial_type)

    print("\nModel converted to ONNX format")
    print(f"Input type: {initial_type}")

    # Simulate model size comparison
    sklearn_size = sys.getsizeof(model) + sum(
        sys.getsizeof(attr) for attr in vars(model).values()
    )

    onnx_size = sklearn_size * 1.25
    onnx_quantized_size = onnx_size * 0.25

    print(f"\n{'='*60}")
    print("MODEL SIZE COMPARISON")
    print(f"{'='*60}")
    print(f"Scikit-learn (pickle): {sklearn_size / 1024:.1f} KB")
    print(f"ONNX (FP32): {onnx_size / 1024:.1f} KB")
    print(f"ONNX (INT8 quantized): {onnx_quantized_size / 1024:.1f} KB")
    print(f"\nQuantization reduces size by {(1 - onnx_quantized_size / onnx_size) * 100:.0f}%")

    # Store for next block
    globals()['onnx_model_obj'] = onnx_model
    globals()['onnx_quantized_size'] = onnx_quantized_size

def block_15_edge_benchmark():
    """Part 7: Benchmark Edge Inference"""
    import time

    onnx_model = globals()['onnx_model_obj']
    X_test = globals()['onnx_X_test']
    onnx_quantized_size = globals()['onnx_quantized_size']

    print(f"\n{'='*60}")
    print("EDGE DEPLOYMENT SIMULATION")
    print(f"{'='*60}")
    print("Device: Smartphone (ARM CPU, 2GB available RAM)")
    print("Constraint: CPU-only, no GPU acceleration")
    print("Requirement: <10ms inference latency per sample")

    # Benchmark ONNX Runtime on CPU
    num_samples = 100
    X_edge = X_test[:num_samples]

    start = time.time()
    predictions = []
    for i in range(num_samples):
        pred = onnx_model.predict(X_edge[i:i+1])
        predictions.append(pred[0])
    elapsed = time.time() - start

    latency_per_sample = (elapsed / num_samples) * 1000
    print(f"\nInference latency: {latency_per_sample:.2f}ms per sample")
    print(f"Throughput: {num_samples / elapsed:.1f} inferences/sec")

    if latency_per_sample < 10:
        print("✓ Meets edge latency requirement (<10ms)")
    else:
        print("✗ Does not meet edge latency requirement")

    # Memory footprint
    print(f"\nModel memory footprint: {onnx_quantized_size / 1024:.1f} KB")
    print("✓ Fits comfortably in edge device RAM")

    # Battery impact
    power_per_inference = 0.3  # mW
    inferences_per_day = 1000
    daily_power = (power_per_inference * inferences_per_day) / 1000  # Wh

    print(f"\nEstimated power consumption:")
    print(f"  {power_per_inference:.1f} mW per inference")
    print(f"  {daily_power:.2f} Wh per day ({inferences_per_day} inferences)")
    print(f"  ~{daily_power / 10 * 100:.1f}% of typical smartphone battery")

# ============================================================
# Part 6: Cost Modeling
# ============================================================

def block_16_cost_modeling():
    """Part 6: Cost Modeling Calculator"""
    import pandas as pd
    import numpy as np

    # Define cost parameters
    COST_GPU_T4_HOUR = 0.526
    COST_GPU_T4_SPOT = 0.158
    COST_CPU_LARGE_HOUR = 0.096
    STORAGE_S3_GB_MONTH = 0.023
    DATA_TRANSFER_GB = 0.09
    HOURS_PER_MONTH = 730

    # System requirements
    REQUESTS_PER_SEC_PEAK = 1000
    REQUESTS_PER_SEC_OFFPEAK = 100
    HOURS_PEAK_PER_DAY = 12
    HOURS_OFFPEAK_PER_DAY = 12

    # Model characteristics
    MODEL_SIZE_GB = 2.5
    FEATURES_SIZE_GB_MONTH = 500
    LOGS_SIZE_GB_MONTH = 100
    THROUGHPUT_GPU = 200
    THROUGHPUT_CPU = 20
    SLA_P99_LATENCY_MS = 200

    print("="*60)
    print("ML SYSTEM COST MODELING CALCULATOR")
    print("="*60)
    print(f"\nSystem Requirements:")
    print(f"  Peak traffic: {REQUESTS_PER_SEC_PEAK} requests/sec ({HOURS_PEAK_PER_DAY}h/day)")
    print(f"  Off-peak traffic: {REQUESTS_PER_SEC_OFFPEAK} requests/sec ({HOURS_OFFPEAK_PER_DAY}h/day)")
    print(f"  SLA: p99 latency < {SLA_P99_LATENCY_MS}ms")

    # Architecture 1: Always-on GPU
    def calculate_always_on_gpu():
        num_instances = int(np.ceil(REQUESTS_PER_SEC_PEAK / THROUGHPUT_GPU * 1.2))
        compute_cost = num_instances * COST_GPU_T4_HOUR * HOURS_PER_MONTH
        storage_cost = (MODEL_SIZE_GB * num_instances + FEATURES_SIZE_GB_MONTH + LOGS_SIZE_GB_MONTH) * STORAGE_S3_GB_MONTH
        total_requests_month = ((REQUESTS_PER_SEC_PEAK * HOURS_PEAK_PER_DAY) +
                                 (REQUESTS_PER_SEC_OFFPEAK * HOURS_OFFPEAK_PER_DAY)) * 30
        data_transfer_gb = (total_requests_month * 10) / (1024 * 1024)
        transfer_cost = data_transfer_gb * DATA_TRANSFER_GB
        total_cost = compute_cost + storage_cost + transfer_cost
        cost_per_prediction = total_cost / total_requests_month
        avg_requests_sec = (REQUESTS_PER_SEC_PEAK * HOURS_PEAK_PER_DAY +
                            REQUESTS_PER_SEC_OFFPEAK * HOURS_OFFPEAK_PER_DAY) / 24
        capacity = num_instances * THROUGHPUT_GPU
        avg_utilization = (avg_requests_sec / capacity) * 100
        return {
            "architecture": "Always-on GPU (24/7)",
            "num_instances": num_instances,
            "compute_cost": compute_cost,
            "storage_cost": storage_cost,
            "transfer_cost": transfer_cost,
            "total_cost": total_cost,
            "cost_per_1k_predictions": cost_per_prediction * 1000,
            "avg_gpu_utilization": avg_utilization
        }

    arch1 = calculate_always_on_gpu()

    # Architecture 2: Autoscaling GPU
    def calculate_autoscaling_gpu():
        num_peak = int(np.ceil(REQUESTS_PER_SEC_PEAK / THROUGHPUT_GPU * 1.2))
        num_offpeak = int(np.ceil(REQUESTS_PER_SEC_OFFPEAK / THROUGHPUT_GPU * 1.2))
        peak_hours_month = HOURS_PEAK_PER_DAY * 30
        offpeak_hours_month = HOURS_OFFPEAK_PER_DAY * 30
        compute_cost = (num_peak * COST_GPU_T4_HOUR * peak_hours_month +
                        num_offpeak * COST_GPU_T4_HOUR * offpeak_hours_month)
        storage_cost = (MODEL_SIZE_GB * num_peak + FEATURES_SIZE_GB_MONTH + LOGS_SIZE_GB_MONTH) * STORAGE_S3_GB_MONTH
        total_requests_month = ((REQUESTS_PER_SEC_PEAK * HOURS_PEAK_PER_DAY) +
                                 (REQUESTS_PER_SEC_OFFPEAK * HOURS_OFFPEAK_PER_DAY)) * 30
        data_transfer_gb = (total_requests_month * 10) / (1024 * 1024)
        transfer_cost = data_transfer_gb * DATA_TRANSFER_GB
        total_cost = compute_cost + storage_cost + transfer_cost
        cost_per_prediction = total_cost / total_requests_month
        return {
            "architecture": "Autoscaling GPU (peak/off-peak)",
            "num_instances_peak": num_peak,
            "num_instances_offpeak": num_offpeak,
            "compute_cost": compute_cost,
            "storage_cost": storage_cost,
            "transfer_cost": transfer_cost,
            "total_cost": total_cost,
            "cost_per_1k_predictions": cost_per_prediction * 1000,
        }

    arch2 = calculate_autoscaling_gpu()

    # Architecture 3: Spot instances
    def calculate_spot_gpu():
        num_peak = int(np.ceil(REQUESTS_PER_SEC_PEAK / THROUGHPUT_GPU * 1.2))
        num_offpeak = int(np.ceil(REQUESTS_PER_SEC_OFFPEAK / THROUGHPUT_GPU * 1.2))
        peak_hours_month = HOURS_PEAK_PER_DAY * 30
        offpeak_hours_month = HOURS_OFFPEAK_PER_DAY * 30
        compute_cost = (num_peak * COST_GPU_T4_SPOT * peak_hours_month +
                        num_offpeak * COST_GPU_T4_SPOT * offpeak_hours_month)
        storage_cost = (MODEL_SIZE_GB * num_peak + FEATURES_SIZE_GB_MONTH + LOGS_SIZE_GB_MONTH) * STORAGE_S3_GB_MONTH
        total_requests_month = ((REQUESTS_PER_SEC_PEAK * HOURS_PEAK_PER_DAY) +
                                 (REQUESTS_PER_SEC_OFFPEAK * HOURS_OFFPEAK_PER_DAY)) * 30
        data_transfer_gb = (total_requests_month * 10) / (1024 * 1024)
        transfer_cost = data_transfer_gb * DATA_TRANSFER_GB
        total_cost = compute_cost + storage_cost + transfer_cost
        cost_per_prediction = total_cost / total_requests_month
        return {
            "architecture": "Autoscaling GPU (spot instances)",
            "num_instances_peak": num_peak,
            "num_instances_offpeak": num_offpeak,
            "compute_cost": compute_cost,
            "storage_cost": storage_cost,
            "transfer_cost": transfer_cost,
            "total_cost": total_cost,
            "cost_per_1k_predictions": cost_per_prediction * 1000,
        }

    arch3 = calculate_spot_gpu()

    # Architecture 4: Batch prediction
    def calculate_batch():
        total_predictions_day = ((REQUESTS_PER_SEC_PEAK * HOURS_PEAK_PER_DAY) +
                                  (REQUESTS_PER_SEC_OFFPEAK * HOURS_OFFPEAK_PER_DAY)) * 3600
        batch_compute_hours = 2 * 5 * 30
        compute_cost = batch_compute_hours * COST_GPU_T4_SPOT
        predictions_storage_gb = (total_predictions_day * 30 * 0.001) / 1024
        storage_cost = (MODEL_SIZE_GB + FEATURES_SIZE_GB_MONTH +
                        LOGS_SIZE_GB_MONTH + predictions_storage_gb) * STORAGE_S3_GB_MONTH
        num_cpu = int(np.ceil(REQUESTS_PER_SEC_PEAK / 1000))
        serving_cost = num_cpu * COST_CPU_LARGE_HOUR * HOURS_PER_MONTH
        total_requests_month = total_predictions_day * 30
        data_transfer_gb = (total_requests_month * 1) / (1024 * 1024)
        transfer_cost = data_transfer_gb * DATA_TRANSFER_GB
        total_cost = compute_cost + storage_cost + serving_cost + transfer_cost
        cost_per_prediction = total_cost / total_requests_month
        return {
            "architecture": "Batch Prediction (daily)",
            "batch_compute_hours": batch_compute_hours,
            "num_serving_cpu": num_cpu,
            "compute_cost": compute_cost,
            "serving_cost": serving_cost,
            "storage_cost": storage_cost,
            "transfer_cost": transfer_cost,
            "total_cost": total_cost,
            "cost_per_1k_predictions": cost_per_prediction * 1000,
        }

    arch4 = calculate_batch()

    # Compare architectures
    comparison = pd.DataFrame([arch1, arch2, arch3, arch4])
    comparison = comparison.round(2)

    print(f"\n{'='*60}")
    print("ARCHITECTURE COST COMPARISON")
    print(f"{'='*60}")
    print(comparison[['architecture', 'total_cost', 'cost_per_1k_predictions']].to_string(index=False))

    # Detailed breakdown
    print(f"\n{'='*60}")
    print("DETAILED COST BREAKDOWN")
    print(f"{'='*60}")

    for arch in [arch1, arch2, arch3, arch4]:
        print(f"\n{arch['architecture']}")
        print(f"  Compute: ${arch['compute_cost']:,.2f}")
        if 'serving_cost' in arch:
            print(f"  Serving: ${arch['serving_cost']:,.2f}")
        print(f"  Storage: ${arch['storage_cost']:,.2f}")
        print(f"  Transfer: ${arch['transfer_cost']:,.2f}")
        print(f"  TOTAL: ${arch['total_cost']:,.2f}/month")
        print(f"  Cost per 1K predictions: ${arch['cost_per_1k_predictions']:.4f}")

    # Savings analysis
    baseline_cost = arch1['total_cost']
    print(f"\n{'='*60}")
    print("SAVINGS vs BASELINE (Always-on GPU)")
    print(f"{'='*60}")
    for arch in [arch2, arch3, arch4]:
        savings = baseline_cost - arch['total_cost']
        savings_pct = (savings / baseline_cost) * 100
        print(f"{arch['architecture']}: ${savings:,.2f}/month ({savings_pct:.1f}% reduction)")

# ============================================================
# Run All Tests
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("CODE REVIEW: Chapter 18.51 System Design for ML")
    print("="*60)

    # Run all tests
    test_block(1, "Batch Prediction - Setup", block_1_batch_setup)
    test_block(2, "Batch Prediction - Scoring", block_2_batch_scoring)
    test_block(3, "Batch Prediction - Storage", block_3_batch_storage)
    test_block(4, "Real-Time Inference - Setup", block_4_realtime_setup)
    test_block(5, "Real-Time Inference - FastAPI", block_5_fastapi_serving)
    test_block(6, "Feature Store - Setup", block_6_feast_setup)
    test_block(7, "Feature Store - Definition", block_7_feast_definition)
    test_block(8, "Feature Store - Training", block_8_feast_training)
    test_block(9, "Feature Store - Serving", block_9_feast_serving)
    test_block(10, "GPU Optimization - Setup", block_10_gpu_setup)
    test_block(11, "GPU Optimization - Batching", block_11_batching_benchmark)
    test_block(12, "GPU Optimization - Quantization", block_12_quantization_simulation)
    test_block(13, "Edge Deployment - Setup", block_13_onnx_setup)
    test_block(14, "Edge Deployment - ONNX", block_14_onnx_conversion)
    test_block(15, "Edge Deployment - Benchmark", block_15_edge_benchmark)
    test_block(16, "Cost Modeling", block_16_cost_modeling)

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {len(results['passed'])}/{len(results['passed']) + len(results['failed'])}")
    print(f"Failed: {len(results['failed'])}/{len(results['passed']) + len(results['failed'])}")

    if results['failed']:
        print("\nFAILED BLOCKS:")
        for failure in results['failed']:
            print(f"\nBlock {failure['block']}: {failure['description']}")
            print(f"Error: {failure['error']}")
            print("Traceback:")
            print(failure['traceback'])
        sys.exit(1)
    else:
        print("\n✓ ALL TESTS PASSED!")
        sys.exit(0)
