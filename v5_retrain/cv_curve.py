"""Per-fold validation accuracy by epoch for the chosen config (same folds/seeds as train_eval.py)."""
import os, sys, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf
sys.argv = [sys.argv[0]] + sys.argv[1:5]
src = open(os.path.join(os.path.dirname(__file__), "train_eval.py")).read()
src = src.split("# ---- 2. cross-validated grid search")[0]          # data, groups, helpers only
exec(src)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(cv.split(np.zeros(N), y_tr, groups))
accs = []
for k, (tr, va) in enumerate(folds):
    tf.keras.utils.set_random_seed(SEED + k)
    h = np.array(fit_epochs(head(0.001, 0.2), tr, np.random.default_rng(SEED + k), EPOCHS, va))
    accs.append(h[:, 1].tolist())
json.dump(accs, open(os.path.join(out_dir, "cv_curve_lr0.001_d0.2.json"), "w"))
a = np.array(accs); print("mean", a.mean(0).round(4).tolist()); print("epoch26", a[:, 25].mean().round(4), a[:, 25].std().round(4))
