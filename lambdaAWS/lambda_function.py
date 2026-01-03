import onnxruntime as ort
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('resnet50', target_size=(224, 224))

session = ort.InferenceSession(
    "sc-model.onnx", providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

classes = [
    "benign",
    "malignant"
]


def predict(url):
    X = preprocessor.from_url(url)
    result = session.run([output_name], {input_name: X})
    float_predictions = result[0][0].tolist()
    return dict(zip(classes, float_predictions))


def handler(event, context):
    url = event["url"]
    result = predict(url)
    return result

