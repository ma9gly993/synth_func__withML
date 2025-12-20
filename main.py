import warnings
import pandas as pd
import optuna.logging
import os
import glob
from data_loader import load_dataset
from models import ModelOptimizer
from metrics import calculate_metrics

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.makedirs("results", exist_ok=True)

def process_dataset(filepath, optimizer):
    dataset_name = os.path.basename(filepath)
    print(f"\nОбработка: {dataset_name}")
    
    if "monks" in filepath and "test" in filepath:
        return None
    
    data = load_dataset(filepath)
    if data is None:
        return None
    
    X_train, X_test, y_train, y_test = data
    
    results = []
    
    is_monks = "monks" in filepath
    
    print(f"  Дерево решений...")
    model, y_pred, params = optimizer.optimize_decision_tree(X_train, y_train, X_test, y_test, is_monks=is_monks)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    metrics = calculate_metrics(y_test, y_pred, y_proba)

    results.append({
        "dataset": dataset_name,
        "model": "DecisionTree",
        "params": str(params),
        **metrics
    })
    
    print(f"  Логистическая регрессия...")
    model, y_pred, params = optimizer.optimize_logistic_regression(X_train, y_train, X_test, y_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    metrics = calculate_metrics(y_test, y_pred, y_proba)

    results.append({
        "dataset": dataset_name,
        "model": "LogisticRegression",
        "params": str(params),
        **metrics
    })
    
    print(f"  Нейросеть...")
    model, y_pred, params = optimizer.optimize_neural_network(X_train, y_train, X_test, y_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    metrics = calculate_metrics(y_test, y_pred, y_proba)

    results.append({
        "dataset": dataset_name,
        "model": "MLPClassifier",
        "params": str(params),
        **metrics
    })
    
    return results

if __name__ == "__main__":
    csv_files = glob.glob("data/*.csv") + glob.glob("data/*.train")
    csv_files = sorted(csv_files)
    
    optimizer = ModelOptimizer(n_trials=50)
    all_results = []
    
    for filepath in csv_files:
        results = process_dataset(filepath, optimizer)
        if results:
            all_results.extend(results)
    
    if all_results:
        df_results = pd.DataFrame(all_results)
        output_path = "results/metrics.csv"
        df_results.to_csv(output_path, index=False)
        print(f"\n✅ Результаты сохранены в {output_path}")
        print(f"\nИтоговая таблица:")
        print(df_results.to_string())
    else:
        print("Нет результатов для сохранения")