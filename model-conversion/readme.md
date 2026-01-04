Converts h5 format to keras or saved-model

# Build
docker build -t h5-converter .

# Convert to keras
docker run -v $(pwd):/models h5-converter model.h5 -o model.keras

# Convert to saved-model
docker run -v $(pwd):/models h5-converter model.h5 --format savedmodel