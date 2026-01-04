docker build -f Dockerfile.keras-onnx -t keras-to-onnx .

docker run -v /workspaces/SkinCancerCapstone/model-conversion:/models keras-to-onnx /models/model.keras -o /models/model.onnx

model.onnx will land in the /to_Onnx directory