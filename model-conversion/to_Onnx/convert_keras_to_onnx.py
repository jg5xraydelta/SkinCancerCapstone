#!/usr/bin/env python3
"""
Keras to ONNX Converter
Converts Keras models (.h5 or .keras) directly to ONNX format
"""

import os
import sys
import argparse


def inspect_keras_model(filepath):
    """Inspect Keras model structure"""
    print(f"\n{'='*60}")
    print(f"Inspecting Keras Model: {filepath}")
    print(f"{'='*60}")
    
    try:
        from tensorflow import keras
        
        model = keras.models.load_model(filepath, compile=False)
        
        print(f"✓ Model loaded successfully")
        print(f"\nModel Summary:")
        model.summary()
        
        print(f"\nInput Shape: {model.input_shape}")
        print(f"Output Shape: {model.output_shape}")
        
    except Exception as e:
        print(f"Error inspecting model: {e}")


def convert_with_tf2onnx(input_path, output_path, opset=13):
    """Convert using tf2onnx from Keras model"""
    print(f"\n[Method 1] Converting with tf2onnx...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Opset: {opset}")
    
    try:
        import tf2onnx
        from tensorflow import keras
        import tensorflow as tf
        
        # Load Keras model
        print("\nLoading Keras model...")
        model = keras.models.load_model(input_path, compile=False)
        print("✓ Model loaded")
        
        # Get input signature
        input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name='input')]
        
        print("Converting to ONNX...")
        # Convert the model
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=opset,
            output_path=output_path
        )
        
        print(f"✓ Conversion successful!")
        print(f"✓ ONNX model saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Method 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_via_saved_model(input_path, output_path, opset=13):
    """Convert by first saving as SavedModel, then to ONNX"""
    print(f"\n[Method 2] Converting via SavedModel intermediate...")
    
    try:
        import tf2onnx
        from tensorflow import keras
        import tempfile
        import shutil
        
        # Load Keras model
        print("\nLoading Keras model...")
        model = keras.models.load_model(input_path, compile=False)
        print("✓ Model loaded")
        
        # Create temporary SavedModel
        temp_dir = tempfile.mkdtemp()
        temp_saved_model = os.path.join(temp_dir, "temp_model")
        
        print(f"Saving as temporary SavedModel...")
        model.export(temp_saved_model)
        print("✓ SavedModel created")
        
        print("Converting SavedModel to ONNX...")
        # Convert SavedModel to ONNX
        onnx_model = tf2onnx.convert.from_saved_model(
            temp_saved_model,
            output_path=output_path,
            opset=opset
        )
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print(f"✓ Conversion successful!")
        print(f"✓ ONNX model saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Method 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_with_keras2onnx(input_path, output_path):
    """Convert using keras2onnx library (alternative)"""
    print(f"\n[Method 3] Converting with keras2onnx...")
    
    try:
        import keras2onnx
        import onnx
        from tensorflow import keras
        
        # Load Keras model
        print("\nLoading Keras model...")
        model = keras.models.load_model(input_path, compile=False)
        print("✓ Model loaded")
        
        print("Converting to ONNX...")
        # Convert
        onnx_model = keras2onnx.convert_keras(model, model.name)
        
        # Save
        onnx.save_model(onnx_model, output_path)
        
        print(f"✓ Conversion successful!")
        print(f"✓ ONNX model saved to: {output_path}")
        
        return True
        
    except ImportError:
        print("✗ keras2onnx not installed (optional library)")
        return False
    except Exception as e:
        print(f"✗ Method 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_onnx_model(onnx_path):
    """Verify the generated ONNX model"""
    print(f"\n{'='*60}")
    print("Verifying ONNX Model")
    print(f"{'='*60}")
    
    try:
        import onnx
        
        # Load and check the model
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        print("✓ ONNX model is valid!")
        
        # Print input/output info
        print("\nModel Inputs:")
        for input_tensor in model.graph.input:
            dims = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in input_tensor.type.tensor_type.shape.dim]
            print(f"  {input_tensor.name}: {dims}")
        
        print("\nModel Outputs:")
        for output_tensor in model.graph.output:
            dims = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in output_tensor.type.tensor_type.shape.dim]
            print(f"  {output_tensor.name}: {dims}")
        
        # Get file size
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"\nModel size: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ ONNX verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert Keras models (.h5 or .keras) to ONNX format'
    )
    parser.add_argument(
        'input',
        help='Input Keras model file (.h5 or .keras)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output ONNX file path (default: model.onnx)',
        default='model.onnx'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=13,
        help='ONNX opset version (default: 13)'
    )
    parser.add_argument(
        '--inspect-only',
        action='store_true',
        help='Only inspect the model without converting'
    )
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='Skip ONNX model verification after conversion'
    )
    parser.add_argument(
        '--method',
        choices=['tf2onnx', 'savedmodel', 'keras2onnx', 'all'],
        default='all',
        help='Conversion method to use (default: all - tries each until one works)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        sys.exit(1)
    
    # Print version info
    print("\nEnvironment Information:")
    print("=" * 60)
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError:
        print("TensorFlow: Not installed")
        sys.exit(1)
    
    try:
        import tf2onnx
        print(f"tf2onnx version: {tf2onnx.__version__}")
    except ImportError:
        print("tf2onnx: Not installed")
        sys.exit(1)
    
    try:
        import onnx
        print(f"ONNX version: {onnx.__version__}")
    except ImportError:
        print("ONNX: Not installed")
    
    try:
        import keras2onnx
        print(f"keras2onnx version: {keras2onnx.__version__}")
    except ImportError:
        print("keras2onnx: Not installed (optional)")
    
    # Inspect the model
    inspect_keras_model(args.input)
    
    if args.inspect_only:
        print("\nInspection complete (--inspect-only flag set)")
        return
    
    # Try conversion methods
    print(f"\n{'='*60}")
    print("Starting Conversion")
    print(f"{'='*60}")
    
    methods = []
    if args.method == 'all':
        methods = [
            ('tf2onnx', lambda: convert_with_tf2onnx(args.input, args.output, args.opset)),
            ('savedmodel', lambda: convert_via_saved_model(args.input, args.output, args.opset))
        ]
    elif args.method == 'tf2onnx':
        methods = [('tf2onnx', lambda: convert_with_tf2onnx(args.input, args.output, args.opset))]
    elif args.method == 'savedmodel':
        methods = [('savedmodel', lambda: convert_via_saved_model(args.input, args.output, args.opset))]
    elif args.method == 'keras2onnx':
        print("Note: keras2onnx is deprecated and not compatible with modern TensorFlow.")
        print("Falling back to tf2onnx method.")
        methods = [('tf2onnx', lambda: convert_with_tf2onnx(args.input, args.output, args.opset))]
    
    success = False
    for method_name, method_func in methods:
        if method_func():
            success = True
            break
    
    if not success:
        print("\n" + "=" * 60)
        print("✗ ALL CONVERSION METHODS FAILED")
        print("=" * 60)
        print("\nSuggestions:")
        print("1. Try different opset versions: --opset 11, --opset 15, or --opset 16")
        print("2. Check if your model uses custom layers")
        print("3. Try simplifying your model architecture")
        print("4. Install keras2onnx: pip install keras2onnx")
        sys.exit(1)
    
    # Verify the ONNX model
    if not args.skip_verify:
        verify_onnx_model(args.output)
    
    print("\n" + "=" * 60)
    print("✓ CONVERSION SUCCESSFUL!")
    print("=" * 60)
    print(f"\nYour ONNX model is ready at: {args.output}")


if __name__ == "__main__":
    main()