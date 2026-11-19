"""Frozen ResNet50 (ImageNet) + global average pooling features.
Train: original + N augmented views (v4 augmentation: shear 10, zoom 0.1, horizontal flip).
Test: original only."""
import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

repo, out, n_aug = sys.argv[1], sys.argv[2], int(sys.argv[3])
limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
# ImageNet ResNet50 base taken from the saved v4 model (frozen during training, so weights are unchanged ImageNet weights)
_v4 = keras.models.load_model(os.path.join(repo, "ResNet_v4_large_15_0.906.h5"), compile=False)
_inp = keras.Input((224, 224, 3))
base = keras.Model(_inp, keras.layers.GlobalAveragePooling2D()(_v4.get_layer("resnet50")(_inp, training=False)))
aug = ImageDataGenerator(shear_range=10.0, zoom_range=0.1, horizontal_flip=True)

def load(split):
    X, y, names = [], [], []
    for label, cls in enumerate(["benign", "malignant"]):
        d = os.path.join(repo, "data", split, cls)
        for f in sorted(os.listdir(d))[:limit]:
            X.append(keras.utils.img_to_array(keras.utils.load_img(os.path.join(d, f), target_size=(224, 224))))
            y.append(label); names.append(f"{cls}/{f}")
    return np.array(X, dtype="uint8"), np.array(y), np.array(names)

def feats(X, augment=False):
    out = []
    for i in range(0, len(X), 256):
        xb = X[i:i+256].astype("float32")
        if augment:
            xb = np.stack([aug.random_transform(x, seed=int(rng.integers(1e9))) for x in xb])
        out.append(base.predict(preprocess_input(xb), batch_size=32, verbose=0))
    return np.concatenate(out)

rng = np.random.default_rng(1)
Xtr, ytr, ntr = load("train"); Xte, yte, nte = load("test")
t = time.time()
F_te = feats(Xte); print("test feats", F_te.shape, f"{time.time()-t:.0f}s", flush=True)
views = [feats(Xtr)]; print("train view 0", f"{time.time()-t:.0f}s", flush=True)
for v in range(n_aug):
    views.append(feats(Xtr, augment=True)); print(f"train view {v+1}", f"{time.time()-t:.0f}s", flush=True)
np.savez_compressed(out, F_tr=np.stack(views), y_tr=ytr, n_tr=ntr, F_te=F_te, y_te=yte, n_te=nte)
print("saved", out)
