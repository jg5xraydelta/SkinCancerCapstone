"""Rebuild the v5 skin-lesion classifier: stock ImageNet ResNet50 base + trained head weights.

The ResNet50 base was frozen during training, so it is Keras's standard ImageNet ResNet50; only the
head's weights (v5_head_weights.npz, under 1 MB, plain NumPy) are stored. Works with Keras 2 and Keras 3.

    python load_v5.py path/to/lesion.jpg
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications import ResNet50


def load_v5(weights_path="v5_head_weights.npz"):
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    head = keras.Sequential([keras.layers.Dense(100, activation="relu"),   # 2048 -> 100
                             keras.layers.Dropout(0.2),
                             keras.layers.Dense(2)], name="head")           # logits [benign, malignant]
    inp = keras.Input((224, 224, 3))
    model = keras.Model(inp, head(keras.layers.GlobalAveragePooling2D()(base(inp, training=False))))
    w = np.load(weights_path)
    head.set_weights([w[f"arr_{i}"] for i in range(len(w.files))])
    return model


if __name__ == "__main__":
    import sys
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input
    model = load_v5()
    img = keras.utils.img_to_array(keras.utils.load_img(sys.argv[1], target_size=(224, 224)))
    p = tf.nn.softmax(model.predict(preprocess_input(img[None]), verbose=0))[0].numpy()
    print(f"benign {p[0]:.3f}  malignant {p[1]:.3f}")
