"""
Code validation script for Chapter 41: Model Interpretability & Explainability
Tests all code blocks in sequence to verify they execute correctly.
"""

import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

def test_block_1_visualization():
    """Test: Interpretability spectrum visualization"""
    print("\n" + "="*70)
    print("TEST 1: Interpretability Spectrum Visualization")
    print("="*70)
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Create interpretability spectrum visualization
        fig, ax = plt.subplots(figsize=(12, 4))

        # Define model categories and positions
        models = [
            'Linear\nRegression',
            'Decision\nTree\n(depth≤5)',
            'Rule\nLists',
            'GAMs',
            'Shallow\nEnsembles',
            'Deep\nDecision Tree',
            'Random\nForest',
            'XGBoost',
            'Deep\nNeural\nNetwork',
            'Large\nTransformer'
        ]
        positions = np.linspace(0, 10, len(models))
        interpretability = [9.5, 8.5, 9.0, 7.5, 6.0, 5.0, 4.0, 3.5, 2.0, 1.0]

        # Color gradient from green (interpretable) to red (black-box)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(models)))

        # Plot bars
        bars = ax.barh(range(len(models)), interpretability, color=colors, alpha=0.8, edgecolor='black')

        # Add model names
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=10)
        ax.set_xlabel('Interpretability Score', fontsize=12, fontweight='bold')
        ax.set_title('The Interpretability Spectrum', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 10)

        # Add vertical regions
        ax.axvspan(7, 10, alpha=0.1, color='green', label='Glass-Box Models')
        ax.axvspan(4, 7, alpha=0.1, color='yellow', label='Moderately Interpretable')
        ax.axvspan(0, 4, alpha=0.1, color='red', label='Black-Box Models')

        # Add annotation
        ax.text(2, -1.5, 'Explainability methods (SHAP, LIME, PDPs) bridge the gap →',
                fontsize=11, style='italic', ha='left', color='navy')

        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig('interpretability_spectrum.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ PASS: Visualization created successfully")
        return True
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        traceback.print_exc()
        return False


def test_block_2_comparing_models():
    """Test: Part 1 - Comparing Interpretable and Black-Box Models"""
    print("\n" + "="*70)
    print("TEST 2: Comparing Interpretable and Black-Box Models")
    print("="*70)
    try:
        import numpy as np
        import pandas as pd
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score

        # Set random seed for reproducibility
        np.random.seed(42)

        # Load breast cancer dataset
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print("Dataset shape:", X_train.shape)
        print("Features (first 5):", list(X_train.columns[:5]))
        print("Target distribution:", pd.Series(y_train).value_counts().to_dict())
        print()

        # Train interpretable model (Logistic Regression)
        lr_model = LogisticRegression(max_iter=5000, random_state=42)
        lr_model.fit(X_train, y_train)
        lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
        lr_auc = roc_auc_score(y_test, lr_pred_proba)

        # Train black-box model (Random Forest)
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_pred_proba)

        print(f"Logistic Regression AUC: {lr_auc:.4f}")
        print(f"Random Forest AUC: {rf_auc:.4f}")
        print(f"Performance difference: {abs(lr_auc - rf_auc):.4f}")
        print()

        # Interpretable model: Examine coefficients
        lr_coefs = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': lr_model.coef_[0]
        }).sort_values('coefficient', ascending=False)

        print("Top 5 features pushing toward MALIGNANT (positive coefficients):")
        print(lr_coefs.head(5).to_string(index=False))
        print()

        print("Top 5 features pushing toward BENIGN (negative coefficients):")
        print(lr_coefs.tail(5).to_string(index=False))

        # Store models for later tests
        global global_rf_model, global_X_train, global_X_test, global_y_train, global_y_test
        global_rf_model = rf_model
        global_X_train = X_train
        global_X_test = X_test
        global_y_train = y_train
        global_y_test = y_test

        print("\n✓ PASS: Models trained and compared successfully")
        return True
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        traceback.print_exc()
        return False


def test_block_3_permutation_importance():
    """Test: Part 2 - Permutation Feature Importance"""
    print("\n" + "="*70)
    print("TEST 3: Permutation Feature Importance")
    print("="*70)
    try:
        from sklearn.inspection import permutation_importance
        import pandas as pd
        import matplotlib.pyplot as plt

        # Use models from previous test
        rf_model = global_rf_model
        X_test = global_X_test
        y_test = global_y_test

        # Compute importance on test set
        perm_importance = permutation_importance(
            rf_model, X_test, y_test,
            n_repeats=10,
            random_state=42,
            scoring='roc_auc'
        )

        # Create DataFrame of results
        perm_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        print("Top 10 features by permutation importance:")
        print(perm_df.head(10).to_string(index=False))
        print()

        # Visualize top 10
        fig, ax = plt.subplots(figsize=(10, 6))
        top_10 = perm_df.head(10)
        ax.barh(range(len(top_10)), top_10['importance_mean'],
                xerr=top_10['importance_std'], color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['feature'])
        ax.set_xlabel('Importance (AUC drop when shuffled)', fontweight='bold')
        ax.set_title('Permutation Feature Importance (Random Forest)', fontweight='bold', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('permutation_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ PASS: Permutation importance computed successfully")
        return True
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        traceback.print_exc()
        return False


def test_block_4_shap_waterfall():
    """Test: Part 3 - SHAP Waterfall Plot"""
    print("\n" + "="*70)
    print("TEST 4: SHAP Waterfall Plot")
    print("="*70)
    try:
        import shap
        import matplotlib.pyplot as plt
        import numpy as np

        rf_model = global_rf_model
        X_test = global_X_test
        y_test = global_y_test

        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(rf_model)

        # Compute SHAP values for test set
        shap_values = explainer.shap_values(X_test)

        # SHAP returns array for each class
        if isinstance(shap_values, list):
            shap_values_class1 = shap_values[1]
        else:
            shap_values_class1 = shap_values

        print(f"SHAP values shape: {shap_values_class1.shape}")
        print(f"Base value (expected model output): {explainer.expected_value[1]:.4f}")
        print()

        # Explain a single prediction
        instance_idx = 0
        instance = X_test.iloc[instance_idx]
        instance_shap = shap_values_class1[instance_idx]
        prediction = rf_model.predict_proba(X_test.iloc[[instance_idx]])[0, 1]

        print(f"Instance {instance_idx} prediction: {prediction:.4f} (probability of malignant)")
        print(f"True label: {'Malignant' if y_test.iloc[instance_idx] == 1 else 'Benign'}")
        print()

        # Create waterfall plot
        shap.plots.waterfall(
            shap.Explanation(
                values=instance_shap,
                base_values=explainer.expected_value[1],
                data=instance.values,
                feature_names=instance.index.tolist()
            ),
            max_display=10,
            show=False
        )
        plt.tight_layout()
        plt.savefig('shap_waterfall.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Verify SHAP values sum to prediction
        shap_sum = explainer.expected_value[1] + instance_shap.sum()
        print(f"Base value + SHAP sum = {shap_sum:.4f}")
        print(f"Actual prediction = {prediction:.4f}")
        print(f"Difference: {abs(shap_sum - prediction):.6f} (should be ~0)")

        # Store for next test
        global global_shap_values_class1, global_explainer
        global_shap_values_class1 = shap_values_class1
        global_explainer = explainer

        print("\n✓ PASS: SHAP waterfall plot created successfully")
        return True
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        traceback.print_exc()
        return False


def test_block_5_shap_summary():
    """Test: Part 4 - SHAP Summary Plot"""
    print("\n" + "="*70)
    print("TEST 5: SHAP Summary Plot")
    print("="*70)
    try:
        import shap
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        shap_values_class1 = global_shap_values_class1
        X_test = global_X_test

        # Create beeswarm summary plot
        shap.summary_plot(
            shap_values_class1,
            X_test,
            max_display=15,
            show=False
        )
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Compute mean absolute SHAP values
        shap_importance = pd.DataFrame({
            'feature': X_test.columns,
            'mean_abs_shap': np.abs(shap_values_class1).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)

        print("Top 10 features by mean |SHAP value|:")
        print(shap_importance.head(10).to_string(index=False))

        print("\n✓ PASS: SHAP summary plot created successfully")
        return True
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        traceback.print_exc()
        return False


def test_block_6_lime_text():
    """Test: Part 5 - LIME Text Explanation"""
    print("\n" + "="*70)
    print("TEST 6: LIME Text Explanation")
    print("="*70)
    try:
        from lime.lime_text import LimeTextExplainer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.pipeline import Pipeline
        from sklearn.datasets import fetch_20newsgroups

        # Load dataset
        categories = ['comp.graphics', 'sci.med']
        newsgroups_train = fetch_20newsgroups(
            subset='train', categories=categories, random_state=42, remove=('headers', 'footers', 'quotes')
        )
        newsgroups_test = fetch_20newsgroups(
            subset='test', categories=categories, random_state=42, remove=('headers', 'footers', 'quotes')
        )

        # Create text classification pipeline
        text_clf = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('clf', LR(max_iter=1000, random_state=42))
        ])

        text_clf.fit(newsgroups_train.data, newsgroups_train.target)

        # Evaluate
        test_preds = text_clf.predict(newsgroups_test.data)
        test_acc = (test_preds == newsgroups_test.target).mean()
        print(f"Text classifier accuracy: {test_acc:.4f}")
        print(f"Classes: {newsgroups_train.target_names}")
        print()

        # Select a test document
        doc_idx = 10
        test_doc = newsgroups_test.data[doc_idx]
        true_label = newsgroups_test.target[doc_idx]
        prediction = text_clf.predict([test_doc])[0]
        pred_proba = text_clf.predict_proba([test_doc])[0]

        print(f"Document {doc_idx} (first 300 chars):")
        print(test_doc[:300] + "...")
        print()
        print(f"True label: {newsgroups_train.target_names[true_label]}")
        print(f"Predicted: {newsgroups_train.target_names[prediction]} (prob={pred_proba[prediction]:.4f})")
        print()

        # Create LIME explainer
        explainer_text = LimeTextExplainer(class_names=newsgroups_train.target_names, random_state=42)

        # Explain the prediction
        explanation = explainer_text.explain_instance(
            test_doc,
            text_clf.predict_proba,
            num_features=10,
            num_samples=500
        )

        # Display explanation
        print("LIME Explanation (top features):")
        for feature, weight in explanation.as_list():
            direction = "→ comp.graphics" if weight < 0 else "→ sci.med"
            print(f"  '{feature}': {weight:+.4f} {direction}")

        # Save HTML visualization
        explanation.save_to_file('lime_text_explanation.html')
        print("\nFull explanation saved to 'lime_text_explanation.html'")

        print("\n✓ PASS: LIME text explanation created successfully")
        return True
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        traceback.print_exc()
        return False


def test_block_7_pdp_ice():
    """Test: Part 6 - Partial Dependence and ICE Plots"""
    print("\n" + "="*70)
    print("TEST 7: Partial Dependence and ICE Plots")
    print("="*70)
    try:
        from sklearn.datasets import fetch_california_housing
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.inspection import PartialDependenceDisplay
        from sklearn.model_selection import train_test_split
        import pandas as pd
        import matplotlib.pyplot as plt

        # Load data
        housing = fetch_california_housing()
        X_house = pd.DataFrame(housing.data, columns=housing.feature_names)
        y_house = housing.target

        # Train/test split
        X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
            X_house, y_house, test_size=0.3, random_state=42
        )

        # Train model
        gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        gb_model.fit(X_h_train, y_h_train)
        gb_score = gb_model.score(X_h_test, y_h_test)

        print(f"Gradient Boosting R² score: {gb_score:.4f}")
        print(f"Feature names: {list(X_house.columns)}")
        print()

        # Create PDP + ICE plot
        fig, ax = plt.subplots(figsize=(10, 6))

        disp = PartialDependenceDisplay.from_estimator(
            gb_model,
            X_h_test,
            features=['MedInc'],
            kind='both',
            ice_lines_kw={'alpha': 0.1, 'linewidth': 0.5},
            pd_line_kw={'color': 'red', 'linewidth': 3, 'label': 'PDP (average)'},
            ax=ax,
            random_state=42,
            subsample=200
        )

        ax.set_ylabel('Predicted House Value ($100k)', fontweight='bold')
        ax.set_xlabel('MedInc (Median Income, $10k)', fontweight='bold')
        ax.set_title('Partial Dependence (PDP) and Individual Conditional Expectation (ICE)',
                     fontweight='bold', fontsize=14)
        ax.legend(['ICE (individual)', 'PDP (average)'], loc='upper left')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('pdp_ice_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ PASS: PDP and ICE plots created successfully")
        return True
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        traceback.print_exc()
        return False


def test_block_8_dice_counterfactuals():
    """Test: Part 7 - Counterfactual Explanations with DiCE"""
    print("\n" + "="*70)
    print("TEST 8: Counterfactual Explanations with DiCE")
    print("="*70)
    try:
        import dice_ml
        import pandas as pd
        import numpy as np

        rf_model = global_rf_model
        X_train = global_X_train
        X_test = global_X_test
        y_train = global_y_train
        y_test = global_y_test

        # Prepare data for DiCE
        dice_data = dice_ml.Data(
            dataframe=pd.concat([X_train, pd.Series(y_train, name='target')], axis=1),
            continuous_features=X_train.columns.tolist(),
            outcome_name='target'
        )

        # Create DiCE model wrapper
        dice_model = dice_ml.Model(model=rf_model, backend='sklearn', model_type='classifier')

        # Create DiCE explainer
        dice_explainer = dice_ml.Dice(dice_data, dice_model, method='random')

        # Select a malignant patient
        malignant_idx = np.where((rf_model.predict(X_test) == 1) & (y_test == 1))[0][0]
        query_instance = X_test.iloc[[malignant_idx]]
        query_pred = rf_model.predict(query_instance)[0]
        query_proba = rf_model.predict_proba(query_instance)[0, 1]

        print(f"Query instance (index {malignant_idx}):")
        print(f"  Predicted: {'Malignant' if query_pred == 1 else 'Benign'} (prob={query_proba:.4f})")
        print(f"  True label: {'Malignant' if y_test.iloc[malignant_idx] == 1 else 'Benign'}")
        print()

        # Generate counterfactuals
        counterfactuals = dice_explainer.generate_counterfactuals(
            query_instance,
            total_CFs=3,
            desired_class=0
        )

        print("Counterfactual Explanations:")
        print("(Minimal changes to flip prediction from Malignant → Benign)\n")
        cf_df = counterfactuals.cf_examples_list[0].final_cfs_df
        print(cf_df.head())

        # Show feature changes
        original_features = query_instance.iloc[0]
        print("\nFeature changes required:")
        for cf_idx in range(min(len(cf_df) - 1, 3)):
            print(f"\n  Counterfactual {cf_idx + 1}:")
            cf = cf_df.iloc[cf_idx]
            changes = []
            for feat in X_test.columns:
                if abs(cf[feat] - original_features[feat]) > 0.01:
                    changes.append(f"{feat}: {original_features[feat]:.2f} → {cf[feat]:.2f}")
            for change in changes[:5]:
                print(f"    {change}")

        print("\n✓ PASS: DiCE counterfactuals generated successfully")
        return True
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        traceback.print_exc()
        return False


def test_solutions():
    """Test all solution code blocks"""
    print("\n" + "="*70)
    print("TEST 9: Solution Code Blocks")
    print("="*70)

    results = []

    # Solution 1: Wine dataset
    try:
        print("\n--- Solution 1: Wine Dataset ---")
        import numpy as np
        import pandas as pd
        from sklearn.datasets import load_wine
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        import shap
        import matplotlib.pyplot as plt

        wine = load_wine()
        X = pd.DataFrame(wine.data, columns=wine.feature_names)
        y = wine.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)

        perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
        perm_df = pd.DataFrame({
            'feature': X.columns,
            'perm_importance': perm_imp.importances_mean
        }).sort_values('perm_importance', ascending=False)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)
        shap_abs = np.abs(shap_values).mean(axis=0).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature': X.columns,
            'shap_importance': shap_abs
        }).sort_values('shap_importance', ascending=False)

        print("Top 5 by Permutation:", list(perm_df.head(5)['feature']))
        print("Top 5 by SHAP:", list(shap_df.head(5)['feature']))
        results.append(("Solution 1", True))
        print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        results.append(("Solution 1", False))

    # Solution 2: Diabetes PDP
    try:
        print("\n--- Solution 2: Diabetes PDP/ICE ---")
        from sklearn.datasets import load_diabetes
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.inspection import PartialDependenceDisplay
        import matplotlib.pyplot as plt

        diabetes = load_diabetes()
        X_diab = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        y_diab = diabetes.target

        gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        gb.fit(X_diab, y_diab)

        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(
            gb, X_diab, features=['bmi'],
            kind='both',
            ice_lines_kw={'alpha': 0.05, 'linewidth': 0.5, 'color': 'gray'},
            pd_line_kw={'color': 'red', 'linewidth': 3},
            ax=ax, subsample=200, random_state=42
        )
        plt.close()

        results.append(("Solution 2", True))
        print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        results.append(("Solution 2", False))

    # Solution 4: Text classification with LIME
    try:
        print("\n--- Solution 4: Text Classification with LIME ---")
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from lime.lime_text import LimeTextExplainer

        categories = ['rec.sport.baseball', 'talk.politics.guns']
        train_data = fetch_20newsgroups(subset='train', categories=categories, random_state=42,
                                        remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(subset='test', categories=categories, random_state=42,
                                       remove=('headers', 'footers', 'quotes'))

        clf = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english')),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ])
        clf.fit(train_data.data, train_data.target)

        preds = clf.predict(test_data.data)
        misclassified_idx = np.where(preds != test_data.target)[0]
        if len(misclassified_idx) > 0:
            doc = test_data.data[misclassified_idx[0]]
            lime_exp = LimeTextExplainer(class_names=train_data.target_names, random_state=42)
            exp = lime_exp.explain_instance(doc, clf.predict_proba, num_features=10, num_samples=500)
            print(f"Explained misclassified document with {len(exp.as_list())} features")

        results.append(("Solution 4", True))
        print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        results.append(("Solution 4", False))

    # Solution 5: Iris with correlated features
    try:
        print("\n--- Solution 5: Iris with Correlated Features ---")
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance

        iris = load_iris()
        X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
        y_iris = iris.target

        rf_iris = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_iris.fit(X_iris, y_iris)

        perm_imp_orig = permutation_importance(rf_iris, X_iris, y_iris, n_repeats=10, random_state=42)
        perm_df_orig = pd.DataFrame({
            'feature': X_iris.columns,
            'importance': perm_imp_orig.importances_mean
        }).sort_values('importance')

        lowest_feat = perm_df_orig.iloc[0]['feature']

        # Add correlated feature
        X_iris_corr = X_iris.copy()
        X_iris_corr[f'{lowest_feat}_correlated'] = (
            0.9 * X_iris[lowest_feat] + 0.1 * np.random.randn(len(X_iris))
        )

        rf_iris_corr = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_iris_corr.fit(X_iris_corr, y_iris)
        perm_imp_corr = permutation_importance(rf_iris_corr, X_iris_corr, y_iris, n_repeats=10, random_state=42)

        print(f"Lowest importance feature: {lowest_feat}")
        results.append(("Solution 5", True))
        print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        results.append(("Solution 5", False))

    # Summary
    print("\n" + "="*70)
    print("SOLUTION TESTS SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")

    return all(passed for _, passed in results)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CHAPTER 41: MODEL INTERPRETABILITY & EXPLAINABILITY")
    print("CODE VALIDATION TEST SUITE")
    print("="*70)

    results = {}

    # Run all tests
    results['visualization'] = test_block_1_visualization()
    results['comparing_models'] = test_block_2_comparing_models()
    results['permutation'] = test_block_3_permutation_importance()
    results['shap_waterfall'] = test_block_4_shap_waterfall()
    results['shap_summary'] = test_block_5_shap_summary()
    results['lime_text'] = test_block_6_lime_text()
    results['pdp_ice'] = test_block_7_pdp_ice()
    results['dice'] = test_block_8_dice_counterfactuals()
    results['solutions'] = test_solutions()

    # Final summary
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<50} {status}")

    print("="*70)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("="*70)

    sys.exit(0 if passed == total else 1)
