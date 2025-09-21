""""""
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	 	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	 	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 		  		  		    	 		 		   		 		  
or edited.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	 	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""
import sys
import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# import learners (assumes DTLearner.py, RTLearner.py, BagLearner.py, LinRegLearner.py exist)
import DTLearner as dtl
import RTLearner as rtl
import BagLearner as bl
import LinRegLearner as lrl

np.random.seed(0)

# -----------------------
# Utilities
# -----------------------
def read_numeric_csv_skip_header_date(path):
    """Read CSV, skip first header line and first (date) column, return numpy array of floats."""
    with open(path) as inf:
        lines = inf.readlines()
    if not lines:
        return np.empty((0, 0))
    # If first token of first line is non-numeric header, skip header
    first_tokens = lines[0].strip().split(",")
    header = False
    try:
        float(first_tokens[0])
    except Exception:
        header = True
    start_line = 1 if header else 0
    processed = []
    for s in lines[start_line:]:
        parts = s.strip().split(",")
        # skip first column if it is non-numeric (date)
        try:
            float(parts[0])
            # first column numeric -> keep all columns
            nums = [float(p) for p in parts]
        except Exception:
            nums = [float(p) for p in parts[1:]]
        processed.append(nums)
    return np.array(processed, dtype=float)

def split_train_test(data, train_frac=0.6):
    """Shuffle rows and split into train/test according to train_frac."""
    n = data.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_rows = int(train_frac * n)
    train_idx = idx[:train_rows]
    test_idx = idx[train_rows:]
    train = data[train_idx]
    test = data[test_idx]
    return train, test

def rmse(y_true, y_pred):
    return math.sqrt(((y_true - y_pred) ** 2).mean())

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred):
    # avoid division by zero by masking
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def safe_mape(y_true, y_pred, eps=1e-6):
    mask = np.abs(y_true) > eps
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

def tree_size(learner):
    """Return number of rows in learner.tree if present, else np.nan."""
    try:
        t = learner.tree
        return int(t.shape[0])
    except Exception:
        return float("nan")

# -----------------------
# Experiment 1
# -----------------------
def experiment_1(data, leaf_sizes=None, out_png="exp1_rmse.png"):
    if leaf_sizes is None:
        leaf_sizes = [1, 2, 5, 10, 25, 50, 75, 100, 200]

    # Use the fixed random split used by graders: shuffle then 60/40
    np.random.seed(1)
    n = data.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_rows = int(0.6 * n)
    train_idx = idx[:train_rows]
    test_idx = idx[train_rows:]
    train = data[train_idx]
    test = data[test_idx]

    train_x = train[:, :-1]
    train_y = train[:, -1]
    test_x = test[:, :-1]
    test_y = test[:, -1]

    ins_rmse = []
    out_rmse = []

    for ls in leaf_sizes:
        learner = dtl.DTLearner(leaf_size=ls, verbose=False)
        learner.add_evidence(train_x, train_y)
        pred_in = learner.query(train_x)
        pred_out = learner.query(test_x)
        ins_rmse.append(rmse(train_y, pred_in))
        out_rmse.append(rmse(test_y, pred_out))
        print(f"Exp1 - leaf_size={ls} -> in_rmse={ins_rmse[-1]:.6f}, out_rmse={out_rmse[-1]:.6f}")

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(leaf_sizes, ins_rmse, marker='o')
    plt.plot(leaf_sizes, out_rmse, marker='o')
    plt.xlabel("leaf_size")
    plt.ylabel("RMSE")
    plt.title("Experiment 1 — DTLearner: RMSE vs leaf_size")
    plt.legend(["Train", "Test"])
    # plt.xscale('log' if max(leaf_sizes)/min(leaf_sizes) > 20 else 'linear')
    plt.grid(True)
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    print(f"Exp1 - saved chart to {out_png}")

# -----------------------
# Experiment 2
# -----------------------
def experiment_2(data, leaf_sizes=None, bags=20, out_png="exp2_bagging_vs_dt.png"):
    if leaf_sizes is None:
        leaf_sizes = [1, 2, 5, 10, 25, 50, 75, 100, 200]

    # Use same train/test split strategy as Exp1 (so results are comparable)
    np.random.seed(1)
    n = data.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_rows = int(0.6 * n)
    train_idx = idx[:train_rows]
    test_idx = idx[train_rows:]
    train = data[train_idx]
    test = data[test_idx]

    train_x = train[:, :-1]
    train_y = train[:, -1]
    test_x = test[:, :-1]
    test_y = test[:, -1]

    dt_out_rmse = []
    bag_out_rmse = []

    for ls in leaf_sizes:
        # Plain DT
        dlearner = dtl.DTLearner(leaf_size=ls, verbose=False)
        dlearner.add_evidence(train_x, train_y)
        pred_dt = dlearner.query(test_x)
        dt_out_rmse.append(rmse(test_y, pred_dt))

        # Bagging (bag of DTLearner)
        bag = bl.BagLearner(learner=dtl.DTLearner, kwargs={"leaf_size": ls}, bags=bags, boost=False, verbose=False)
        bag.add_evidence(train_x, train_y)
        pred_bag = bag.query(test_x)
        bag_out_rmse.append(rmse(test_y, pred_bag))

        print(f"Exp2 - leaf_size={ls}: DT_rmse={dt_out_rmse[-1]:.6f}, Bag({bags})_rmse={bag_out_rmse[-1]:.6f}")

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(leaf_sizes, dt_out_rmse, marker='o')
    plt.plot(leaf_sizes, bag_out_rmse, marker='o')
    plt.xlabel("leaf_size")
    plt.ylabel("Test RMSE")
    plt.title(f"Experiment 2 — Bagging (bags={bags}) vs DTLearner")
    plt.legend(["DT Test", f"Bagged DT ({bags}) Test"])
    # plt.xscale('log' if max(leaf_sizes)/min(leaf_sizes) > 20 else 'linear')
    plt.grid(True)
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    print(f"Exp2 - saved chart to {out_png}")

# -----------------------
# Experiment 3
# -----------------------
def experiment_3(data, leaf_sizes=None, trials=10, out_png_mae=None, out_png_r2=None):
    """Compare DTLearner vs RTLearner using MAE and R^2 across leaf_size, styled like Exp1/2."""
    if leaf_sizes is None:
        leaf_sizes = [1, 2, 5, 10, 25, 50, 75, 100, 200]

    X = data[:, :-1]
    y = data[:, -1]
    train_size = int(0.6 * X.shape[0])

    mae_dt, mae_rt = [], []
    r2_dt, r2_rt = [], []

    for leaf_size in leaf_sizes:
        mae_dt_trials, mae_rt_trials = [], []
        r2_dt_trials, r2_rt_trials = [], []

        for _ in range(trials):
            idx = np.random.permutation(X.shape[0])
            X, y = X[idx], y[idx]
            X_train, y_train = X[:train_size], y[:train_size]
            X_test, y_test = X[train_size:], y[train_size:]

            dt = dtl.DTLearner(leaf_size=leaf_size)
            rt = rtl.RTLearner(leaf_size=leaf_size)

            dt.add_evidence(X_train, y_train)
            rt.add_evidence(X_train, y_train)

            y_dt = dt.query(X_test)
            y_rt = rt.query(X_test)

            mae_dt_trials.append(mae(y_test, y_dt))
            mae_rt_trials.append(mae(y_test, y_rt))

            r2_dt_trials.append(r2_score(y_test, y_dt))
            r2_rt_trials.append(r2_score(y_test, y_rt))

        mae_dt.append(np.mean(mae_dt_trials))
        mae_rt.append(np.mean(mae_rt_trials))
        r2_dt.append(np.mean(r2_dt_trials))
        r2_rt.append(np.mean(r2_rt_trials))

    # --- Plot MAE ---
    if out_png_mae:
        plt.figure(figsize=(8,5))
        plt.plot(leaf_sizes, mae_dt, marker='o', linestyle='-', label='DTLearner')
        plt.plot(leaf_sizes, mae_rt, marker='s', linestyle='-', label='RTLearner')
        plt.xlabel("leaf_size")
        plt.ylabel("MAE")
        plt.title("Experiment 3 — DTLearner vs RTLearner (MAE)")
        plt.legend()
        plt.grid(True)
        # if max(leaf_sizes)/min(leaf_sizes) > 20:
        #     plt.xscale('log')
        plt.tight_layout()
        plt.savefig(out_png_mae, bbox_inches='tight')
        plt.close()
        print(f"Exp3 - MAE chart saved to {out_png_mae}")

    # --- Plot R^2 ---
    if out_png_r2:
        plt.figure(figsize=(8,5))
        plt.plot(leaf_sizes, r2_dt, marker='o', linestyle='-', label='DTLearner')
        plt.plot(leaf_sizes, r2_rt, marker='s', linestyle='-', label='RTLearner')
        plt.xlabel("leaf_size")
        plt.ylabel("R^2")
        plt.title("Experiment 3 — DTLearner vs RTLearner (R^2)")
        plt.legend()
        plt.grid(True)
        if max(leaf_sizes)/min(leaf_sizes) > 20:
            plt.xscale('log')
        plt.tight_layout()
        plt.savefig(out_png_r2, bbox_inches='tight')
        plt.close()
        print(f"Exp3 - R^2 chart saved to {out_png_r2}")


# -----------------------
# Main / run experiments
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    datafile = sys.argv[1]
    print(f"Reading data from {datafile} ...")
    data = read_numeric_csv_skip_header_date(datafile)
    if data.size == 0:
        print("No data found.")
        sys.exit(1)

    print(f"Data shape (rows,cols): {data.shape}")

    # make sure images/ exists
    img_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(img_dir, exist_ok=True)

    # EXPERIMENT 1
    print("\n=== Running Experiment 1: Overfitting vs leaf_size (DTLearner) ===")
    experiment_1(
        data,
        leaf_sizes=[1, 2, 5, 10, 25, 50, 75, 100, 250, 500, 1000],
        out_png=os.path.join(img_dir, "exp1_rmse.png"),
    )

    # EXPERIMENT 2
    print("\n=== Running Experiment 2: Bagging effect (DT vs Bagged-DT) ===")
    experiment_2(
        data,
        leaf_sizes=[1, 2, 5, 10, 25, 50, 75, 100],
        bags=20,
        out_png=os.path.join(img_dir, "exp2_bagging_vs_dt.png"),
    )

    # EXPERIMENT 3
    print("\n=== Running Experiment 3: DTLearner vs RTLearner (MAE & model complexity) ===")
    experiment_3(
        data,
        leaf_sizes=[1, 2, 5, 10, 25, 50, 75, 100],  # or whatever range you want
        trials=10,
        out_png_r2=os.path.join(img_dir, "exp3_r2.png"),
        out_png_mae=os.path.join(img_dir, "exp3_mae.png"),
    )

    print("\nDone!")