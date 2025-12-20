import warnings
import numpy as np
import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from metrics import get_f1_average

warnings.filterwarnings('ignore')

class BestTrialCallback:
    def __init__(self):
        self.best_value = None
    
    def __call__(self, study, trial):
        if self.best_value is None or trial.value > self.best_value:
            self.best_value = trial.value
            print(f"    Trial {trial.number}: F1 = {trial.value:.4f} (лучший)")

class ModelOptimizer:
    def __init__(self, n_trials=50):
        self.n_trials = n_trials
    
    def optimize_decision_tree(self, X_train, y_train, X_test, y_test, is_monks=False):

        def objective(trial):
            # --- MONK’s: жёсткие, правильные ограничения
            if is_monks:
                max_depth = trial.suggest_int("max_depth", 2, 6)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 5)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 2)
                criterion = "entropy"
                max_features = None
            else:
                max_depth = trial.suggest_int("max_depth", 1, 50)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 50)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
                criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
                max_features = trial.suggest_categorical(
                    "max_features", [None, "sqrt", "log2"]
                )

            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                max_features=max_features,
                random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # --- ЖЁСТКИЙ ШТРАФ ЗА ВЫРОЖДЕНИЕ
            if len(np.unique(y_pred)) == 1:
                return -1.0

            return f1_score(y_test, y_pred, average="binary")

        study = optuna.create_study(direction="maximize")
        callback = BestTrialCallback()
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False, callbacks=[callback])

        best_params = study.best_params

        # MONK’s: параметры не трогаем
        if is_monks:
            best_params["criterion"] = "entropy"
            best_params["max_features"] = None

        model = DecisionTreeClassifier(**best_params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return model, y_pred, best_params
    
    def optimize_logistic_regression(self, X_train, y_train, X_test, y_test):
        def objective(trial):
            C = trial.suggest_float("C", 0.01, 100, log=True)
            penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None])
            solver = "lbfgs"
            l1_ratio = None
            if penalty == "l1":
                solver = "liblinear"
            elif penalty == "elasticnet":
                solver = "saga"
                l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            
            max_iter = trial.suggest_int("max_iter", 100, 1000)
            
            try:
                model_params = {
                    "C": C,
                    "penalty": penalty,
                    "solver": solver,
                    "max_iter": max_iter,
                    "random_state": 42
                }
                if l1_ratio is not None:
                    model_params["l1_ratio"] = l1_ratio
                
                model = LogisticRegression(**model_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                average = get_f1_average(y_test)
                if average == "binary":
                    unique_labels = np.unique(y_test)
                    pos_label = 1 if 1 in unique_labels else unique_labels[1] if len(unique_labels) == 2 else None
                    if pos_label is not None:
                        return f1_score(y_test, y_pred, average=average, pos_label=pos_label)
                return f1_score(y_test, y_pred, average=average)
            except:
                return 0.0
        
        study = optuna.create_study(direction="maximize")
        callback = BestTrialCallback()
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False, callbacks=[callback])
        
        best_params = study.best_params
        penalty = best_params["penalty"]
        solver = "lbfgs"
        l1_ratio = None
        if penalty == "l1":
            solver = "liblinear"
        elif penalty == "elasticnet":
            solver = "saga"
            l1_ratio = best_params.get("l1_ratio")
        
        model_params = {
            "C": best_params["C"],
            "penalty": penalty,
            "solver": solver,
            "max_iter": best_params["max_iter"],
            "random_state": 42
        }
        if l1_ratio is not None:
            model_params["l1_ratio"] = l1_ratio
        
        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        return model, y_pred, best_params
    
    def optimize_neural_network(self, X_train, y_train, X_test, y_test):
        def objective(trial):
            hidden_layer_sizes = tuple([
                trial.suggest_int(f"n_neurons_l{i}", 10, 200) 
                for i in range(trial.suggest_int("n_layers", 1, 3))
            ])
            alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
            learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)
            max_iter = trial.suggest_int("max_iter", 100, 500)
            
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred = np.asarray(y_pred, dtype=np.int32)
                average = get_f1_average(y_test)
                if average == "binary":
                    unique_labels = np.unique(y_test)
                    pos_label = 1 if 1 in unique_labels else unique_labels[1] if len(unique_labels) == 2 else None
                    if pos_label is not None:
                        return f1_score(y_test, y_pred, average=average, pos_label=pos_label)
                return f1_score(y_test, y_pred, average=average)
            except:
                return 0.0
        
        study = optuna.create_study(direction="maximize")
        callback = BestTrialCallback()
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False, callbacks=[callback])
        
        best_params = study.best_params
        n_layers = best_params["n_layers"]
        hidden_layer_sizes = tuple([
            best_params[f"n_neurons_l{i}"] 
            for i in range(n_layers)
        ])
        
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=best_params["alpha"],
            learning_rate_init=best_params["learning_rate_init"],
            max_iter=best_params["max_iter"],
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.asarray(y_pred, dtype=np.int32)
        
        return model, y_pred, best_params
