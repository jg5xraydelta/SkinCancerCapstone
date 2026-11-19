"""Leak-free model selection and a single final test evaluation.

1. Group near-duplicate training images (same lesion photographed more than once) so a
   lesion never sits on both sides of a train/validation split.
2. 5-fold stratified group cross-validation on the TRAINING set only, over the v4 head
   (Dense 100 ReLU -> Dropout -> Dense 2), grid: learning rate x dropout x epoch.
3. Retrain the chosen configuration on all training images, then score the test set once:
   all 660 images, and the 636 left after removing 24 that duplicate a training lesion.
"""
import json, os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import imagehash
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

repo, feat_path, dup_path, out_dir = sys.argv[1:5]
EPOCHS, SEED = int(os.environ.get('EPOCHS', 30)), 1
LRS, DROPS = [0.01, 0.001, 0.0001], [0.0, 0.2, 0.5, 0.8]

d = np.load(feat_path)
F_tr, y_tr, n_tr, F_te, y_te, n_te = d["F_tr"], d["y_tr"], d["n_tr"], d["F_te"], d["y_te"], d["n_te"]
V, N = F_tr.shape[0], F_tr.shape[1]

# ---- 1. lesion groups inside the training set (perceptual hash <= 4, same label) ----
bits = np.array([imagehash.phash(Image.open(os.path.join(repo, "data/train", n))).hash.flatten() for n in n_tr])
parent = list(range(N))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
for i in range(N):
    dist = (bits[i + 1:] != bits[i]).sum(1)
    for j in np.nonzero((dist <= 4) & (y_tr[i + 1:] == y_tr[i]))[0] + i + 1:
        parent[find(i)] = find(j)
groups = np.array([find(i) for i in range(N)])
print(f"train images {N}, lesion groups {len(set(groups))}", flush=True)

# ---- test images that duplicate a training lesion (from dupcheck, visually confirmed) ----
NOT_SAME = {"benign/1.jpg", "benign/1074.jpg", "benign/1473.jpg", "benign/216.jpg"}  # look-alikes, different lesions
dups = json.load(open(dup_path))["near"]
leaked = {f"{tc}/{tf}" for tc, tf, rc, rf, dist in dups if tc == rc and f"{tc}/{tf}" not in NOT_SAME}
clean = np.array([n not in leaked for n in n_te])
print(f"test images {len(n_te)}, leaked {len(leaked)}, clean {clean.sum()}", flush=True)

def head(lr, drop):
    m = keras.Sequential([keras.Input((2048,)), keras.layers.Dense(100, activation="relu"),
                          keras.layers.Dropout(drop), keras.layers.Dense(2)])
    m.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    return m

def fit_epochs(m, idx, rng, epochs, val=None):
    hist = []
    for _ in range(epochs):
        v = rng.integers(V, size=len(idx))                 # a fresh augmented view per image each epoch
        m.fit(F_tr[v, idx], y_tr[idx], batch_size=32, epochs=1, verbose=0, shuffle=True)
        if val is not None:
            hist.append(m.evaluate(F_tr[0, val], y_tr[val], verbose=0))  # validation: un-augmented view
    return hist

# ---- 2. cross-validated grid search on training data only ----
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(cv.split(np.zeros(N), y_tr, groups))
grid = {}
for lr in LRS:
    for drop in DROPS:
        accs, losses = [], []
        for k, (tr, va) in enumerate(folds):
            tf.keras.utils.set_random_seed(SEED + k)
            h = np.array(fit_epochs(head(lr, drop), tr, np.random.default_rng(SEED + k), EPOCHS, va))
            losses.append(h[:, 0]); accs.append(h[:, 1])
        grid[(lr, drop)] = (np.array(accs), np.array(losses))
        a = np.array(accs).mean(0)
        print(f"lr={lr} drop={drop}: best mean val acc {a.max():.4f} at epoch {a.argmax()+1}", flush=True)

best = max(((lr, dr, e) for (lr, dr) in grid for e in range(EPOCHS)),
           key=lambda t: (grid[t[:2]][0][:, t[2]].mean(), -grid[t[:2]][1][:, t[2]].mean()))
lr, drop, ep = best[0], best[1], best[2] + 1
cv_acc = grid[(lr, drop)][0][:, ep - 1]
print(f"CHOSEN lr={lr} dropout={drop} epochs={ep}; CV acc {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}", flush=True)

# ---- 3. final model on all training images; single test evaluation ----
tf.keras.utils.set_random_seed(SEED)
final = head(lr, drop)
fit_epochs(final, np.arange(N), np.random.default_rng(SEED), ep)
probs = tf.nn.softmax(final.predict(F_te, verbose=0), axis=1).numpy()[:, 1]

def metrics(y, p, rng=np.random.default_rng(SEED), B=2000):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    f = lambda yy, pp: ((pp >= .5) == yy).mean(),
    def stats(yy, pp):
        pr = (pp >= .5).astype(int)
        return np.array([(pr == yy).mean(), pr[yy == 1].mean(), 1 - pr[yy == 0].mean(), roc_auc_score(yy, pp)])
    boot = []
    for _ in range(B):
        i = rng.integers(len(y), size=len(y))
        boot.append(stats(y[i], p[i]))
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    point = stats(y, p)
    names = ["accuracy", "sensitivity", "specificity", "auc"]
    r = {n: [float(point[k]), float(lo[k]), float(hi[k])] for k, n in enumerate(names)}
    r.update(n=int(len(y)), cm=[[int(tn), int(fp)], [int(fn), int(tp)]], ppv=float(tp / (tp + fp)))
    return r

res = {"chosen": {"lr": lr, "dropout": drop, "epochs": ep},
       "cv_acc_mean": float(cv_acc.mean()), "cv_acc_sd": float(cv_acc.std()),
       "n_groups": int(len(set(groups))), "leaked": sorted(leaked),
       "test_all": metrics(y_te, probs), "test_clean": metrics(y_te[clean], probs[clean]),
       "curves": {f"{l}|{dr}": grid[(l, dr)][0].mean(0).round(4).tolist() for (l, dr) in grid}}
fpr, tpr, _ = roc_curve(y_te[clean], probs[clean])
res["roc_clean"] = [fpr.round(4).tolist(), tpr.round(4).tolist()]
json.dump(res, open(os.path.join(out_dir, "results_v5.json"), "w"), indent=1)
np.save(os.path.join(out_dir, "test_probs_v5.npy"), probs)

# full deployable model: frozen base + new head
v4 = keras.models.load_model(os.path.join(repo, "ResNet_v4_large_15_0.906.h5"), compile=False)
inp = keras.Input((224, 224, 3))
x = keras.layers.GlobalAveragePooling2D()(v4.get_layer("resnet50")(inp, training=False))
full = keras.Model(inp, final(x))
full.save(os.path.join(out_dir, f"ResNet_v5_cv_{ep:02d}_{res['test_clean']['accuracy'][0]:.3f}.keras"))
print(json.dumps({k: res[k] for k in ["chosen", "cv_acc_mean", "cv_acc_sd", "test_all", "test_clean"]}, indent=1))
