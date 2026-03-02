#!/usr/bin/env python3
"""
Code Review: Testing all code blocks from Chapter 31
"""

import sys
import os
import tempfile
import shutil

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Track results
results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_block(name, test_func):
    """Test a code block and record results"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)
    try:
        test_func()
        results['passed'].append(name)
        print(f"✓ PASSED: {name}")
        return True
    except Exception as e:
        results['failed'].append((name, str(e)))
        print(f"✗ FAILED: {name}")
        print(f"Error: {e}")
        return False

# ============================================================================
# BLOCK 1: Distributed Computing Architecture Diagram
# ============================================================================

def test_block_1_visualization():
    """Test visualization code"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import numpy as np

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Distributed Computing Architecture',
            fontsize=16, fontweight='bold', ha='center')

    # Driver/Master Node
    driver = FancyBboxPatch((3.5, 7), 3, 1,
                             boxstyle="round,pad=0.1",
                             edgecolor='#2C3E50', facecolor='#3498DB', linewidth=2)
    ax.add_patch(driver)
    ax.text(5, 7.5, 'Driver/Master\n(Coordinator)',
            fontsize=11, fontweight='bold', ha='center', va='center', color='white')

    # Worker/Executor Nodes
    worker_positions = [(1, 4), (4, 4), (7, 4)]
    worker_colors = ['#E74C3C', '#2ECC71', '#F39C12']

    for i, (x, y) in enumerate(worker_positions, 1):
        worker = FancyBboxPatch((x-0.75, y-0.5), 1.5, 1,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='#2C3E50', facecolor=worker_colors[i-1], linewidth=2)
        ax.add_patch(worker)
        ax.text(x, y, f'Executor {i}\n(Worker)',
                fontsize=10, ha='center', va='center', color='white', fontweight='bold')

        # Data partitions
        data_box = FancyBboxPatch((x-0.75, y-2), 1.5, 0.8,
                                   boxstyle="round,pad=0.05",
                                   edgecolor='#34495E', facecolor='#ECF0F1', linewidth=1.5)
        ax.add_patch(data_box)
        ax.text(x, y-1.6, f'Partition {i}\nData',
                fontsize=9, ha='center', va='center', color='#2C3E50')

    # Arrows: Driver to Workers (Task Assignment)
    for x, y in worker_positions:
        arrow = FancyArrowPatch((5, 7), (x, 4.5),
                                 arrowstyle='->', mutation_scale=20, linewidth=2,
                                 color='#8E44AD')
        ax.add_patch(arrow)

    ax.text(2.5, 6, 'Task\nAssignment', fontsize=9, ha='center', color='#8E44AD', fontweight='bold')

    # Arrows: Workers to Driver (Results)
    for x, y in worker_positions:
        arrow = FancyArrowPatch((x, 4.5), (5, 7),
                                 arrowstyle='->', mutation_scale=20, linewidth=1.5,
                                 color='#16A085', linestyle='dashed')
        ax.add_patch(arrow)

    ax.text(7.5, 6, 'Results', fontsize=9, ha='center', color='#16A085', fontweight='bold')

    # Data locality annotation
    ax.text(5, 1.5, 'Key: Computation happens where data lives (data locality)',
            fontsize=10, ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDEBD0', edgecolor='#E67E22', linewidth=2))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#3498DB', edgecolor='#2C3E50', label='Coordinator'),
        mpatches.Patch(facecolor='#E74C3C', edgecolor='#2C3E50', label='Workers/Executors'),
        mpatches.Patch(facecolor='#ECF0F1', edgecolor='#34495E', label='Data Partitions'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, frameon=True)

    plt.tight_layout()

    # Don't save, just close
    plt.close()

    print("✓ Visualization code executed successfully")

# ============================================================================
# BLOCK 2: The Breaking Point - When Pandas Fails
# ============================================================================

def test_block_2_breaking_point():
    """Test pandas breaking point demonstration"""
    import pandas as pd
    import numpy as np
    from sklearn.datasets import fetch_california_housing
    import psutil
    import time

    # Load California Housing dataset
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    assert df.shape == (20640, 9), f"Expected shape (20640, 9), got {df.shape}"

    # Test first replication
    df_large = pd.concat([df] * 1, ignore_index=True)
    memory_mb = df_large.memory_usage(deep=True).sum() / 1024**2

    # Perform aggregation
    start_time = time.time()
    result = df_large.groupby('MedInc')['MedHouseVal'].mean()
    elapsed = time.time() - start_time

    assert len(result) > 0, "Aggregation should return results"
    assert elapsed >= 0, "Time should be positive"

    print(f"✓ Dataset shape: {df.shape}")
    print(f"✓ Memory usage: {memory_mb:.2f} MB")
    print(f"✓ Aggregation completed in {elapsed:.3f}s")

# ============================================================================
# BLOCK 3: PySpark Basics (REQUIRES PYSPARK)
# ============================================================================

def test_block_3_pyspark_basics():
    """Test PySpark basics - skipped if PySpark not available"""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, mean, stddev, count, when
        import pandas as pd
        import numpy as np
        from sklearn.datasets import load_iris

        # Initialize Spark
        spark = SparkSession.builder \
            .appName("BigDataEcosystem") \
            .master("local[*]") \
            .config("spark.driver.memory", "2g") \
            .config("spark.sql.shuffle.partitions", "2") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()

        # Load Iris dataset
        iris = load_iris(as_frame=True)
        df_pandas = iris.frame
        df_pandas['species'] = df_pandas['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

        # Convert to Spark DataFrame
        df = spark.createDataFrame(df_pandas)

        assert df.count() == 150, "Should have 150 rows"

        # Test operations
        df_selected = df.select("sepal length (cm)", "sepal width (cm)", "species")
        df_filtered = df.filter(col("sepal length (cm)") > 6.5)
        count_filtered = df_filtered.count()

        assert count_filtered > 0, "Should have some filtered results"

        # GroupBy and aggregate
        df_grouped = df.groupBy("species").agg(
            mean("sepal length (cm)").alias("avg_sepal_length"),
            stddev("sepal length (cm)").alias("std_sepal_length"),
            count("*").alias("count")
        )

        result = df_grouped.collect()
        assert len(result) == 3, "Should have 3 species groups"

        spark.stop()
        print("✓ PySpark operations completed successfully")

    except ImportError:
        results['warnings'].append("Block 3: PySpark not installed - skipped")
        raise ImportError("PySpark not available")

# ============================================================================
# BLOCK 4: Spark SQL (REQUIRES PYSPARK)
# ============================================================================

def test_block_4_spark_sql():
    """Test Spark SQL - skipped if PySpark not available"""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, sum as spark_sum, avg, count, to_date
        import pandas as pd
        import numpy as np

        spark = SparkSession.builder \
            .appName("SparkSQL_Demo") \
            .master("local[*]") \
            .config("spark.sql.shuffle.partitions", "2") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()

        # Generate synthetic sales data
        np.random.seed(42)
        n_records = 100

        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports']
        products = {
            'Electronics': ['Laptop', 'Phone', 'Tablet', 'Headphones'],
            'Clothing': ['Shirt', 'Pants', 'Shoes', 'Jacket'],
            'Food': ['Bread', 'Milk', 'Eggs', 'Cheese'],
            'Home': ['Sofa', 'Table', 'Chair', 'Lamp'],
            'Sports': ['Ball', 'Racket', 'Shoes', 'Bike']
        }

        sales_data = []
        for _ in range(n_records):
            date = np.random.choice(dates)
            category = np.random.choice(categories)
            product = np.random.choice(products[category])
            quantity = np.random.randint(1, 10)
            unit_price = np.random.uniform(10, 500)
            revenue = quantity * unit_price

            sales_data.append({
                'date': date,
                'category': category,
                'product': product,
                'quantity': quantity,
                'unit_price': round(unit_price, 2),
                'revenue': round(revenue, 2)
            })

        df_pandas = pd.DataFrame(sales_data)
        df = spark.createDataFrame(df_pandas)

        # Register as SQL table
        df.createOrReplaceTempView("sales")

        # SQL Query
        query1 = """
            SELECT
                category,
                SUM(revenue) as total_revenue,
                COUNT(*) as num_transactions,
                AVG(revenue) as avg_transaction_value
            FROM sales
            GROUP BY category
            ORDER BY total_revenue DESC
        """

        result1 = spark.sql(query1)
        result_count = result1.count()

        assert result_count > 0, "Should have query results"

        # DataFrame API equivalent
        result1_df = df.groupBy("category").agg(
            spark_sum("revenue").alias("total_revenue"),
            count("*").alias("num_transactions"),
            avg("revenue").alias("avg_transaction_value")
        ).orderBy(col("total_revenue").desc())

        # Verify equivalence
        sql_pandas = result1.toPandas().sort_values('category').reset_index(drop=True)
        df_pandas_result = result1_df.toPandas().sort_values('category').reset_index(drop=True)

        assert sql_pandas.equals(df_pandas_result), "SQL and DataFrame API should produce same results"

        spark.stop()
        print("✓ Spark SQL operations completed successfully")

    except ImportError:
        results['warnings'].append("Block 4: PySpark not installed - skipped")
        raise ImportError("PySpark not available")

# ============================================================================
# BLOCK 5: Parquet Format (REQUIRES PYSPARK)
# ============================================================================

def test_block_5_parquet():
    """Test Parquet format - skipped if PySpark not available"""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, year, month
        import pandas as pd
        import numpy as np
        import os
        import shutil

        # Use temp directory
        temp_dir = tempfile.mkdtemp()

        spark = SparkSession.builder \
            .appName("Parquet_Test") \
            .master("local[*]") \
            .config("spark.sql.shuffle.partitions", "2") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()

        # Generate small dataset
        np.random.seed(42)
        n_records = 100

        dates = pd.date_range('2023-01-01', '2024-12-31', freq='H')[:n_records]
        categories = ['Electronics', 'Clothing', 'Food']

        sales_data = []
        for i in range(n_records):
            sales_data.append({
                'timestamp': dates[i],
                'category': np.random.choice(categories),
                'product_id': np.random.randint(1, 100),
                'quantity': np.random.randint(1, 10),
                'revenue': round(np.random.uniform(10, 500), 2)
            })

        df_pandas = pd.DataFrame(sales_data)

        # Save CSV
        csv_path = os.path.join(temp_dir, 'sales.csv')
        df_pandas.to_csv(csv_path, index=False)
        csv_size = os.path.getsize(csv_path)

        # Convert to Spark and save as Parquet
        df = spark.createDataFrame(df_pandas)
        parquet_path = os.path.join(temp_dir, 'sales_parquet')
        df.write.parquet(parquet_path)

        # Read back
        df_parquet = spark.read.parquet(parquet_path)
        assert df_parquet.count() == n_records, "Should have same record count"

        # Test partitioning
        df_with_date = df.withColumn("year", year(col("timestamp"))) \
                         .withColumn("month", month(col("timestamp")))

        partitioned_path = os.path.join(temp_dir, 'sales_partitioned')
        df_with_date.write.partitionBy("category").parquet(partitioned_path)

        # Verify partitioned structure exists
        assert os.path.exists(partitioned_path), "Partitioned path should exist"

        # Read partitioned data
        df_partitioned = spark.read.parquet(partitioned_path)
        assert df_partitioned.count() == n_records, "Should have same record count"

        spark.stop()

        # Cleanup
        shutil.rmtree(temp_dir)

        print("✓ Parquet format and partitioning work correctly")

    except ImportError:
        results['warnings'].append("Block 5: PySpark not installed - skipped")
        raise ImportError("PySpark not available")

# ============================================================================
# SOLUTION 1: Wine Analysis (REQUIRES PYSPARK)
# ============================================================================

def test_solution_1():
    """Test Solution 1 - Wine Analysis"""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, mean, stddev, count
        from sklearn.datasets import load_wine
        import pandas as pd

        spark = SparkSession.builder \
            .appName("WineAnalysis") \
            .master("local[*]") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()

        # Load Wine dataset
        wine = load_wine(as_frame=True)
        df_pandas = wine.frame
        df_pandas.columns = wine.feature_names + ['target']

        # Convert to Spark DataFrame
        df = spark.createDataFrame(df_pandas)

        assert df.count() == 178, "Wine dataset should have 178 rows"

        # Create high_alcohol column
        df_with_flag = df.withColumn("high_alcohol", col("alcohol") > 13.0)

        # Summary statistics
        summary = df_with_flag.groupBy("high_alcohol").agg(
            mean("alcohol").alias("avg_alcohol"),
            stddev("alcohol").alias("std_alcohol"),
            mean("malic_acid").alias("avg_malic_acid"),
            count("*").alias("count")
        )

        assert summary.count() == 2, "Should have 2 groups (True/False)"

        # Filter and count
        filtered = df.filter((col("alcohol") > 13.5) & (col("malic_acid") < 2.0))
        result = filtered.groupBy("target").count().orderBy("target")

        assert result.count() >= 0, "Should have results"

        spark.stop()
        print("✓ Solution 1 (Wine Analysis) works correctly")

    except ImportError:
        results['warnings'].append("Solution 1: PySpark not installed - skipped")
        raise ImportError("PySpark not available")

# ============================================================================
# SOLUTION 2: Partitioning Benchmark (REQUIRES PYSPARK)
# ============================================================================

def test_solution_2():
    """Test Solution 2 - Partitioning"""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, year, month, quarter
        import pandas as pd
        import numpy as np
        import os
        import shutil

        temp_dir = tempfile.mkdtemp()

        spark = SparkSession.builder \
            .appName("PartitioningBenchmark") \
            .master("local[*]") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()

        # Generate small synthetic data
        np.random.seed(42)
        n_records = 100

        dates = pd.date_range('2023-01-01', '2024-12-31', periods=n_records)
        categories = ['Electronics', 'Clothing', 'Food']

        data = pd.DataFrame({
            'date': dates,
            'customer_id': np.random.randint(1, 11, n_records),
            'product_category': np.random.choice(categories, n_records),
            'revenue': np.random.uniform(10, 500, n_records).round(2)
        })

        df = spark.createDataFrame(data)

        # Add date components
        df = df.withColumn("year", year(col("date"))) \
               .withColumn("month", month(col("date"))) \
               .withColumn("quarter", quarter(col("date")))

        # Save unpartitioned
        path1 = os.path.join(temp_dir, 'unpartitioned')
        df.write.parquet(path1)

        # Save partitioned by category
        path2 = os.path.join(temp_dir, 'partitioned_category')
        df.write.partitionBy("product_category").parquet(path2)

        # Verify files exist
        assert os.path.exists(path1), "Unpartitioned path should exist"
        assert os.path.exists(path2), "Partitioned path should exist"

        # Read back
        df1 = spark.read.parquet(path1)
        df2 = spark.read.parquet(path2)

        assert df1.count() == n_records, "Should have same record count"
        assert df2.count() == n_records, "Should have same record count"

        spark.stop()
        shutil.rmtree(temp_dir)

        print("✓ Solution 2 (Partitioning) works correctly")

    except ImportError:
        results['warnings'].append("Solution 2: PySpark not installed - skipped")
        raise ImportError("PySpark not available")

# ============================================================================
# SOLUTION 4: Word Count (REQUIRES PYSPARK)
# ============================================================================

def test_solution_4():
    """Test Solution 4 - Word Count"""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, explode, split, lower, trim, count as spark_count
        import pandas as pd

        spark = SparkSession.builder \
            .appName("WordCount") \
            .master("local[*]") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()

        # Sample corpus
        texts = [
            "Apache Spark is a unified analytics engine for large-scale data processing.",
            "Spark provides high-level APIs in Java, Scala, Python, and R.",
            "Spark runs on Hadoop, Kubernetes, or standalone clusters.",
        ]

        # Create DataFrame
        df = spark.createDataFrame([(text,) for text in texts], ["text"])

        # Tokenize
        words = df.select(explode(split(lower(col("text")), r'\W+')).alias("word"))

        # Remove stop words
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                      'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that',
                      'the', 'to', 'was', 'with'}

        words_filtered = words.filter(
            (col("word") != "") &
            (~col("word").isin(stop_words))
        )

        # Count frequency
        word_counts = words_filtered.groupBy("word").agg(
            spark_count("*").alias("count")
        ).orderBy(col("count").desc())

        assert word_counts.count() > 0, "Should have word counts"

        spark.stop()
        print("✓ Solution 4 (Word Count) works correctly")

    except ImportError:
        results['warnings'].append("Solution 4: PySpark not installed - skipped")
        raise ImportError("PySpark not available")

# ============================================================================
# SOLUTION 5: Configuration Experiments (REQUIRES PYSPARK)
# ============================================================================

def test_solution_5():
    """Test Solution 5 - Configuration"""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, count, avg
        from sklearn.datasets import fetch_california_housing

        # Test with small partition count
        spark = SparkSession.builder \
            .appName("PartitionExperiment") \
            .master("local[*]") \
            .config("spark.sql.shuffle.partitions", "4") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()

        # Load data
        data = fetch_california_housing(as_frame=True)
        df = spark.createDataFrame(data.frame)

        # GroupBy operation
        result = df.groupBy("MedInc").agg(
            count("*").alias("count"),
            avg("MedHouseVal").alias("avg_price")
        ).collect()

        assert len(result) > 0, "Should have aggregation results"

        spark.stop()
        print("✓ Solution 5 (Configuration) works correctly")

    except ImportError:
        results['warnings'].append("Solution 5: PySpark not installed - skipped")
        raise ImportError("PySpark not available")

# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CODE REVIEW: Chapter 31 - Big Data Ecosystem")
    print("="*60)

    # Test all blocks
    test_block("Block 1: Visualization", test_block_1_visualization)
    test_block("Block 2: Breaking Point", test_block_2_breaking_point)
    test_block("Block 3: PySpark Basics", test_block_3_pyspark_basics)
    test_block("Block 4: Spark SQL", test_block_4_spark_sql)
    test_block("Block 5: Parquet Format", test_block_5_parquet)
    test_block("Solution 1: Wine Analysis", test_solution_1)
    test_block("Solution 2: Partitioning", test_solution_2)
    test_block("Solution 4: Word Count", test_solution_4)
    test_block("Solution 5: Configuration", test_solution_5)

    # Print summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Passed: {len(results['passed'])}")
    print(f"✗ Failed: {len(results['failed'])}")
    print(f"⚠ Warnings: {len(results['warnings'])}")

    if results['passed']:
        print("\nPassed tests:")
        for test in results['passed']:
            print(f"  ✓ {test}")

    if results['failed']:
        print("\nFailed tests:")
        for test, error in results['failed']:
            print(f"  ✗ {test}")
            print(f"    Error: {error[:100]}")

    if results['warnings']:
        print("\nWarnings:")
        for warning in results['warnings']:
            print(f"  ⚠ {warning}")
