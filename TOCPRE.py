import os
import time
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 1. 可选依赖导入
# =========================
HAS_LIGHTGBM = True
HAS_TABPFN = True

try:
    import lightgbm as lgb
except Exception as e:
    HAS_LIGHTGBM = False
    print(f"[WARN] LightGBM 导入失败: {e}")

try:
    from tabpfn import TabPFNRegressor
except Exception as e:
    HAS_TABPFN = False
    print(f"[WARN] TabPFN 导入失败: {e}")


# =========================
# 2. 全局配置
# =========================
DATA_PATH = "./WXN2.csv"
RANDOM_STATE = 420
TEST_SIZE = 0.2
SAVE_DIR = "./all_model_results"

os.makedirs(SAVE_DIR, exist_ok=True)

# 统一PSO参数
PSO_PARAMS = {
    "n_particles": 20,
    "max_iter": 200,
    "c1": 2.05,
    "c2": 2.05,
    "w_min": 0.4,
    "w_max": 0.9
}

# RF 参数范围
RF_PARAM_BOUNDS = {
    "n_estimators": (10, 500),
    "max_depth": (3, 20),
    "min_samples_split": (2, 10),
    "min_samples_leaf": (1, 10),
    "max_features": (0.1, 1.0)
}

# SVM 参数范围
SVM_PARAM_BOUNDS = {
    "C": (0.1, 100),
    "gamma": (0.001, 10),
    "epsilon": (0.01, 1)
}

# LightGBM 参数范围
LGB_PARAM_BOUNDS = {
    "learning_rate": (0.01, 0.3),
    "n_estimators": (10, 500),
    "max_depth": (3, 10),
    "feature_fraction": (0.5, 1.0),
    "bagging_fraction": (0.5, 1.0)
}


# =========================
# 3. 工具函数
# =========================
def evaluate_regression(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mse, rmse, mae


def save_predictions(y_true, y_pred, model_name):
    pred_df = pd.DataFrame({
        "True_TOC": np.asarray(y_true).ravel(),
        "Predicted_TOC": np.asarray(y_pred).ravel()
    })
    pred_path = os.path.join(SAVE_DIR, f"{model_name}_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    return pred_path


def plot_true_vs_pred(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    line_min = min(np.min(y_true), np.min(y_pred))
    line_max = max(np.max(y_true), np.max(y_pred))
    plt.plot([line_min, line_max], [line_min, line_max], "k--", lw=2)
    plt.xlabel("True TOC")
    plt.ylabel("Predicted TOC")
    plt.title(f"{model_name}: True vs Predicted")
    plt.grid(True)
    fig_path = os.path.join(SAVE_DIR, f"{model_name}_true_vs_pred.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_fitness_history(history, model_name):
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(history)), history)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (1 - R²)")
    plt.title(f"{model_name} Optimization Process")
    plt.grid(True)
    fig_path = os.path.join(SAVE_DIR, f"{model_name}_optimization.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path


# =========================
# 4. 自定义PSO
# =========================
class CustomPSO:
    def __init__(self, objective_func, bounds, n_particles=20, max_iter=100,
                 c1=2.05, c2=2.05, w_min=0.4, w_max=0.9, random_state=42):
        self.objective_func = objective_func
        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w_min = w_min
        self.w_max = w_max
        self.dim = len(bounds)
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.positions = np.zeros((n_particles, self.dim))
        self.velocities = np.zeros((n_particles, self.dim))

        for i in range(self.dim):
            param_name = list(bounds.keys())[i]
            lower, upper = bounds[param_name]
            self.positions[:, i] = np.random.uniform(lower, upper, n_particles)
            self.velocities[:, i] = np.random.uniform(
                -0.1 * (upper - lower),
                0.1 * (upper - lower),
                n_particles
            )

        self.pbest_positions = self.positions.copy()
        self.pbest_fitness = np.full(n_particles, float("inf"))

        self.gbest_position = np.zeros(self.dim)
        self.gbest_fitness = float("inf")

        self.fitness_history = []

    def update_particles(self, iteration):
        w = self.w_max - (self.w_max - self.w_min) * iteration / self.max_iter

        for i in range(self.n_particles):
            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)

            cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
            social = self.c2 * r2 * (self.gbest_position - self.positions[i])
            self.velocities[i] = w * self.velocities[i] + cognitive + social

            self.positions[i] = self.positions[i] + self.velocities[i]

            for j in range(self.dim):
                param_name = list(self.bounds.keys())[j]
                lower, upper = self.bounds[param_name]
                self.positions[i, j] = np.clip(self.positions[i, j], lower, upper)

    def optimize(self, verbose=True, print_every=50):
        for i in range(self.n_particles):
            fitness = self.objective_func(self.positions[i])
            self.pbest_fitness[i] = fitness
            self.pbest_positions[i] = self.positions[i].copy()

            if fitness < self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_position = self.positions[i].copy()

        self.fitness_history.append(self.gbest_fitness)

        for iteration in range(self.max_iter):
            self.update_particles(iteration)

            for i in range(self.n_particles):
                fitness = self.objective_func(self.positions[i])

                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()

                if fitness < self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = self.positions[i].copy()

            self.fitness_history.append(self.gbest_fitness)

            if verbose and (iteration % print_every == 0 or iteration == self.max_iter - 1):
                print(f"Iteration {iteration}, Best Fitness: {self.gbest_fitness:.6f}")

        return self.gbest_position, self.gbest_fitness


# =========================
# 5. 读取统一数据
# =========================
data = pd.read_csv(DATA_PATH, encoding="ISO-8859-1", sep=",", header=None)
X = data.iloc[:, 0:14].copy()
y = data.iloc[:, 14].copy()

print("=" * 70)
print("Unified dataset loaded")
print(f"Data path: {DATA_PATH}")
print(f"Data shape: {data.shape}")
print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")
print("=" * 70)

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# 给SVM单独做标准化，保持和你原脚本一致
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train_svm = scaler_x.fit_transform(X_train_raw)
X_test_svm = scaler_x.transform(X_test_raw)

y_train_svm = scaler_y.fit_transform(y_train_raw.values.reshape(-1, 1)).ravel()
y_test_svm = scaler_y.transform(y_test_raw.values.reshape(-1, 1)).ravel()


# =========================
# 6. RF
# =========================
def run_pso_rf():
    print("\n" + "=" * 70)
    print("Starting PSO-RF...")
    print("=" * 70)

    def pso_rf_objective(params):
        n_estimators = int(params[0])
        max_depth = int(params[1])
        min_samples_split = int(params[2])
        min_samples_leaf = int(params[3])
        max_features = float(params[4])

        n_estimators = max(RF_PARAM_BOUNDS["n_estimators"][0], min(RF_PARAM_BOUNDS["n_estimators"][1], n_estimators))
        max_depth = max(RF_PARAM_BOUNDS["max_depth"][0], min(RF_PARAM_BOUNDS["max_depth"][1], max_depth))
        min_samples_split = max(RF_PARAM_BOUNDS["min_samples_split"][0], min(RF_PARAM_BOUNDS["min_samples_split"][1], min_samples_split))
        min_samples_leaf = max(RF_PARAM_BOUNDS["min_samples_leaf"][0], min(RF_PARAM_BOUNDS["min_samples_leaf"][1], min_samples_leaf))
        max_features = max(RF_PARAM_BOUNDS["max_features"][0], min(RF_PARAM_BOUNDS["max_features"][1], max_features))

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )

        try:
            model.fit(X_train_raw, y_train_raw)
            y_pred = model.predict(X_test_raw)
            r2 = r2_score(y_test_raw, y_pred)
            return 1 - r2
        except Exception:
            return 10.0

    start_time = time.time()

    pso_optimizer = CustomPSO(
        pso_rf_objective,
        RF_PARAM_BOUNDS,
        **PSO_PARAMS,
        random_state=42
    )
    best_params, best_fitness = pso_optimizer.optimize(verbose=True, print_every=50)

    best_model = RandomForestRegressor(
        n_estimators=int(best_params[0]),
        max_depth=int(best_params[1]),
        min_samples_split=int(best_params[2]),
        min_samples_leaf=int(best_params[3]),
        max_features=float(best_params[4]),
        random_state=42,
        n_jobs=-1
    )
    best_model.fit(X_train_raw, y_train_raw)
    y_pred_train = best_model.predict(X_train_raw)
    y_pred = best_model.predict(X_test_raw)

    elapsed = time.time() - start_time
    r2_train, mse_train, rmse_train, mae_train = evaluate_regression(y_train_raw, y_pred_train)
    r2, mse, rmse, mae = evaluate_regression(y_test_raw, y_pred)

    model_path = os.path.join(SAVE_DIR, "pso_rf_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    pred_path = save_predictions(y_test_raw, y_pred, "PSO_RF")
    fig_pred = plot_true_vs_pred(y_test_raw, y_pred, "PSO_RF")
    fig_opt = plot_fitness_history(pso_optimizer.fitness_history, "PSO_RF")

    return {
        "Model": "PSO-RF",
        "Train_R2": r2_train,
        "Train_MSE": mse_train,
        "Train_RMSE": rmse_train,
        "Train_MAE": mae_train,
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Time_s": elapsed,
        "Best_Params": {
            "n_estimators": int(best_params[0]),
            "max_depth": int(best_params[1]),
            "min_samples_split": int(best_params[2]),
            "min_samples_leaf": int(best_params[3]),
            "max_features": float(best_params[4])
        },
        "Prediction_File": pred_path,
        "Figure_File": fig_pred,
        "Optimization_Figure": fig_opt,
        "Model_File": model_path
    }


# =========================
# 7. SVM
# =========================
def run_pso_svm():
    print("\n" + "=" * 70)
    print("Starting PSO-SVM...")
    print("=" * 70)

    def pso_svm_objective(params):
        C = float(params[0])
        gamma = float(params[1])
        epsilon = float(params[2])

        C = max(SVM_PARAM_BOUNDS["C"][0], min(SVM_PARAM_BOUNDS["C"][1], C))
        gamma = max(SVM_PARAM_BOUNDS["gamma"][0], min(SVM_PARAM_BOUNDS["gamma"][1], gamma))
        epsilon = max(SVM_PARAM_BOUNDS["epsilon"][0], min(SVM_PARAM_BOUNDS["epsilon"][1], epsilon))

        model = SVR(C=C, gamma=gamma, epsilon=epsilon, kernel="rbf")

        try:
            model.fit(X_train_svm, y_train_svm)
            y_pred = model.predict(X_test_svm)
            r2 = r2_score(y_test_svm, y_pred)
            return 1 - r2
        except Exception:
            return 10.0

    start_time = time.time()

    pso_optimizer = CustomPSO(
        pso_svm_objective,
        SVM_PARAM_BOUNDS,
        **PSO_PARAMS,
        random_state=42
    )
    best_params, best_fitness = pso_optimizer.optimize(verbose=True, print_every=50)

    best_model = SVR(
        C=float(best_params[0]),
        gamma=float(best_params[1]),
        epsilon=float(best_params[2]),
        kernel="rbf"
    )
    best_model.fit(X_train_svm, y_train_svm)

    y_pred_train_scaled = best_model.predict(X_train_svm)
    y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
    y_train_original = y_train_raw.values

    y_pred_scaled = best_model.predict(X_test_svm)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_original = y_test_raw.values

    elapsed = time.time() - start_time
    r2_train, mse_train, rmse_train, mae_train = evaluate_regression(y_train_original, y_pred_train)
    r2, mse, rmse, mae = evaluate_regression(y_test_original, y_pred)

    model_path = os.path.join(SAVE_DIR, "pso_svm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    scaler_x_path = os.path.join(SAVE_DIR, "svm_scaler_x.pkl")
    scaler_y_path = os.path.join(SAVE_DIR, "svm_scaler_y.pkl")
    with open(scaler_x_path, "wb") as f:
        pickle.dump(scaler_x, f)
    with open(scaler_y_path, "wb") as f:
        pickle.dump(scaler_y, f)

    pred_path = save_predictions(y_test_original, y_pred, "PSO_SVM")
    fig_pred = plot_true_vs_pred(y_test_original, y_pred, "PSO_SVM")
    fig_opt = plot_fitness_history(pso_optimizer.fitness_history, "PSO_SVM")

    return {
        "Model": "PSO-SVM",
        "Train_R2": r2_train,
        "Train_MSE": mse_train,
        "Train_RMSE": rmse_train,
        "Train_MAE": mae_train,
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Time_s": elapsed,
        "Best_Params": {
            "C": float(best_params[0]),
            "gamma": float(best_params[1]),
            "epsilon": float(best_params[2])
        },
        "Prediction_File": pred_path,
        "Figure_File": fig_pred,
        "Optimization_Figure": fig_opt,
        "Model_File": model_path,
        "Scaler_X_File": scaler_x_path,
        "Scaler_Y_File": scaler_y_path
    }


# =========================
# 8. LightGBM
# =========================
def run_pso_lgb():
    if not HAS_LIGHTGBM:
        return {
            "Model": "PSO-LightGBM",
            "Train_R2": np.nan,
            "Train_MSE": np.nan,
            "Train_RMSE": np.nan,
            "Train_MAE": np.nan,
            "R2": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Time_s": np.nan,
            "Best_Params": "LightGBM not installed"
        }

    print("\n" + "=" * 70)
    print("Starting PSO-LightGBM...")
    print("=" * 70)

    def pso_lgb_objective(params):
        learning_rate = float(params[0])
        n_estimators = int(params[1])
        max_depth = int(params[2])
        feature_fraction = float(params[3])
        bagging_fraction = float(params[4])

        learning_rate = max(LGB_PARAM_BOUNDS["learning_rate"][0], min(LGB_PARAM_BOUNDS["learning_rate"][1], learning_rate))
        n_estimators = max(LGB_PARAM_BOUNDS["n_estimators"][0], min(LGB_PARAM_BOUNDS["n_estimators"][1], n_estimators))
        max_depth = max(LGB_PARAM_BOUNDS["max_depth"][0], min(LGB_PARAM_BOUNDS["max_depth"][1], max_depth))
        feature_fraction = max(LGB_PARAM_BOUNDS["feature_fraction"][0], min(LGB_PARAM_BOUNDS["feature_fraction"][1], feature_fraction))
        bagging_fraction = max(LGB_PARAM_BOUNDS["bagging_fraction"][0], min(LGB_PARAM_BOUNDS["bagging_fraction"][1], bagging_fraction))

        model = lgb.LGBMRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=1,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )

        try:
            model.fit(X_train_raw, y_train_raw)
            y_pred = model.predict(X_test_raw)
            r2 = r2_score(y_test_raw, y_pred)
            return 1 - r2
        except Exception:
            return 10.0

    start_time = time.time()

    pso_optimizer = CustomPSO(
        pso_lgb_objective,
        LGB_PARAM_BOUNDS,
        **PSO_PARAMS,
        random_state=42
    )
    best_params, best_fitness = pso_optimizer.optimize(verbose=True, print_every=50)

    best_model = lgb.LGBMRegressor(
        learning_rate=float(best_params[0]),
        n_estimators=int(best_params[1]),
        max_depth=int(best_params[2]),
        feature_fraction=float(best_params[3]),
        bagging_fraction=float(best_params[4]),
        bagging_freq=1,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    best_model.fit(X_train_raw, y_train_raw)
    y_pred_train = best_model.predict(X_train_raw)
    y_pred = best_model.predict(X_test_raw)

    elapsed = time.time() - start_time
    r2_train, mse_train, rmse_train, mae_train = evaluate_regression(y_train_raw, y_pred_train)
    r2, mse, rmse, mae = evaluate_regression(y_test_raw, y_pred)

    model_path = os.path.join(SAVE_DIR, "pso_lgb_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    pred_path = save_predictions(y_test_raw, y_pred, "PSO_LightGBM")
    fig_pred = plot_true_vs_pred(y_test_raw, y_pred, "PSO_LightGBM")
    fig_opt = plot_fitness_history(pso_optimizer.fitness_history, "PSO_LightGBM")

    return {
        "Model": "PSO-LightGBM",
        "Train_R2": r2_train,
        "Train_MSE": mse_train,
        "Train_RMSE": rmse_train,
        "Train_MAE": mae_train,
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Time_s": elapsed,
        "Best_Params": {
            "learning_rate": float(best_params[0]),
            "n_estimators": int(best_params[1]),
            "max_depth": int(best_params[2]),
            "feature_fraction": float(best_params[3]),
            "bagging_fraction": float(best_params[4])
        },
        "Prediction_File": pred_path,
        "Figure_File": fig_pred,
        "Optimization_Figure": fig_opt,
        "Model_File": model_path
    }


# =========================
# 9. TabPFN
# =========================
def run_tabpfn():
    if not HAS_TABPFN:
        return {
            "Model": "TabPFN",
            "Train_R2": np.nan,
            "Train_MSE": np.nan,
            "Train_RMSE": np.nan,
            "Train_MAE": np.nan,
            "R2": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Time_s": np.nan,
            "Best_Params": "TabPFN not installed"
        }

    print("\n" + "=" * 70)
    print("Starting TabPFN...")
    print("=" * 70)

    start_time = time.time()

    # 如数据特别大且报预训练限制，可改为：
    # model = TabPFNRegressor(ignore_pretraining_limits=True)
    model = TabPFNRegressor()
    model.fit(X_train_raw, y_train_raw)
    y_pred_train = model.predict(X_train_raw)
    y_pred = model.predict(X_test_raw)

    elapsed = time.time() - start_time
    r2_train, mse_train, rmse_train, mae_train = evaluate_regression(y_train_raw, y_pred_train)
    r2, mse, rmse, mae = evaluate_regression(y_test_raw, y_pred)

    model_path = os.path.join(SAVE_DIR, "tabpfn_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    pred_path = save_predictions(y_test_raw, y_pred, "TabPFN")
    fig_pred = plot_true_vs_pred(y_test_raw, y_pred, "TabPFN")

    return {
        "Model": "TabPFN",
        "Train_R2": r2_train,
        "Train_MSE": mse_train,
        "Train_RMSE": rmse_train,
        "Train_MAE": mae_train,
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Time_s": elapsed,
        "Best_Params": "Default TabPFNRegressor()",
        "Prediction_File": pred_path,
        "Figure_File": fig_pred,
        "Optimization_Figure": "",
        "Model_File": model_path
    }


# =========================
# 10. 主程序
# =========================
if __name__ == "__main__":
    all_results = []

    # 按顺序运行
    for runner in [run_pso_rf, run_pso_svm, run_pso_lgb, run_tabpfn]:
        try:
            result = runner()
            all_results.append(result)

            print("\nResult Summary:")
            print(f"Model: {result['Model']}")
            print(f"Train_R2: {result['Train_R2']:.4f}")
            print(f"Train_MSE: {result['Train_MSE']:.4f}")
            print(f"Train_RMSE: {result['Train_RMSE']:.4f}")
            print(f"Train_MAE: {result['Train_MAE']:.4f}")
            print(f"Test_R2: {result['R2']:.4f}")
            print(f"Test_MSE: {result['MSE']:.4f}")
            print(f"Test_RMSE: {result['RMSE']:.4f}")
            print(f"Test_MAE: {result['MAE']:.4f}")
            print(f"Time_s: {result['Time_s']:.2f}")
            print(f"Best_Params: {result['Best_Params']}")
        except Exception as e:
            print(f"[ERROR] {runner.__name__} failed: {e}")
            all_results.append({
                "Model": runner.__name__,
                "Train_R2": np.nan,
                "Train_MSE": np.nan,
                "Train_RMSE": np.nan,
                "Train_MAE": np.nan,
                "R2": np.nan,
                "MSE": np.nan,
                "RMSE": np.nan,
                "MAE": np.nan,
                "Time_s": np.nan,
                "Best_Params": f"Failed: {e}"
            })

    # 汇总结果表
    summary_rows = []
    detail_rows = []

    for item in all_results:
        summary_rows.append({
            "Model": item.get("Model"),
            "Train_R2": item.get("Train_R2"),
            "Train_MSE": item.get("Train_MSE"),
            "Train_RMSE": item.get("Train_RMSE"),
            "Train_MAE": item.get("Train_MAE"),
            "Test_R2": item.get("R2"),
            "Test_MSE": item.get("MSE"),
            "Test_RMSE": item.get("RMSE"),
            "Test_MAE": item.get("MAE"),
            "Time_s": item.get("Time_s")
        })

        detail_rows.append({
            "Model": item.get("Model"),
            "Best_Params": json.dumps(item.get("Best_Params", ""), ensure_ascii=False),
            "Prediction_File": item.get("Prediction_File", ""),
            "Figure_File": item.get("Figure_File", ""),
            "Optimization_Figure": item.get("Optimization_Figure", ""),
            "Model_File": item.get("Model_File", "")
        })

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    summary_path = os.path.join(SAVE_DIR, "model_summary.csv")
    detail_path = os.path.join(SAVE_DIR, "model_details.csv")

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("All models finished.")
    print("=" * 70)
    print(summary_df)

    # 画训练集 vs 测试集 R² 对比图
    valid_summary = summary_df.dropna(subset=["Test_R2"]).copy()
    if len(valid_summary) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(valid_summary))
        width = 0.35
        ax.bar(x - width/2, valid_summary["Train_R2"], width, label="Train R²")
        ax.bar(x + width/2, valid_summary["Test_R2"], width, label="Test R²")
        ax.set_xlabel("Model")
        ax.set_ylabel("R²")
        ax.set_title("Train vs Test R² Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(valid_summary["Model"], rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y")
        comp_fig = os.path.join(SAVE_DIR, "model_comparison_r2.png")
        plt.savefig(comp_fig, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\nSummary saved to: {summary_path}")
    print(f"Details saved to: {detail_path}")
    print(f"All outputs are in folder: {SAVE_DIR}")