"""Independent check: load the saved full v5 model, run it on raw test images, compare to reported results."""
import json, os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import confusion_matrix, roc_auc_score
repo, model_path, res_path = sys.argv[1:4]
m = keras.models.load_model(model_path)
res = json.load(open(res_path)); leaked = set(res["leaked"])
X, y, keep = [], [], []
for lab, cls in enumerate(["benign", "malignant"]):
    d = os.path.join(repo, "data/test", cls)
    for f in sorted(os.listdir(d)):
        X.append(keras.utils.img_to_array(keras.utils.load_img(os.path.join(d, f), target_size=(224, 224))))
        y.append(lab); keep.append(f"{cls}/{f}" not in leaked)
X, y, keep = preprocess_input(np.array(X)), np.array(y), np.array(keep)
p = tf.nn.softmax(m.predict(X, batch_size=32, verbose=0), axis=1).numpy()[:, 1]
for name, k in [("all", np.ones_like(keep)), ("clean", keep)]:
    yy, pp = y[k.astype(bool)], p[k.astype(bool)]; pr = (pp >= .5).astype(int)
    cm = confusion_matrix(yy, pr).tolist()
    print(name, len(yy), "acc", round((pr == yy).mean(), 4), "auc", round(roc_auc_score(yy, pp), 4), "cm", cm,
          "matches report:", cm == res["test_" + name]["cm"])
