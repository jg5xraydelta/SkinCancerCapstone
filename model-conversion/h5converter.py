#!/usr/bin/env python3
"""
H5 Model Converter Script
Converts Keras H5 models to .keras format (for latest TensorFlow)
"""

import os
import sys
import argparse
import h5py


def inspect_h5_file(filepath):
    """Inspect the structure of an H5 file"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*60}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"Top-level keys: {list(f.keys())}")
            
            if 'model_weights' in f.keys():
                print("Type: Weights-only file")
            else:
                print("Type: Full model file")
    except Exception as e:
        print(f"Error inspecting file: {e}")


def convert_standard(input_path, output_path):
    """Standard conversion for latest TensorFlow"""
    print("\n[Attempt 1] Standard load (latest TensorFlow)...")
    
    try:
        from tensorflow import keras
        
        model = keras.models.load_model(input_path, compile=False)
        print("✓ Model loaded successfully!")
        
        model.save(output_path)
        print(f"✓ Model saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def convert_with_safe_mode_off(input_path, output_path):
    """Try conversion with safe_mode=False for Keras 3"""
    print("\n[Attempt 2] Load with safe_mode=False...")
    
    try:
        from tensorflow import keras
        
        model = keras.models.load_model(input_path, compile=False, safe_mode=False)
        print("✓ Model loaded successfully with safe_mode=False!")
        
        model.save(output_path)
        print(f"✓ Model saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def convert_to_savedmodel(input_path, output_path):
    """Convert to TensorFlow SavedModel format"""
    print("\n[Attempt 3] Converting to SavedModel format...")
    
    try:
        from tensorflow import keras
        
        model = keras.models.load_model(input_path, compile=False)
        print("✓ Model loaded successfully!")
        
        # Remove .keras or .h5 extension and use as directory name
        output_dir = output_path.replace('.keras', '').replace('.h5', '') + '_savedmodel'
        model.export(output_dir)
        print(f"✓ Model exported to SavedModel: {output_dir}")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def convert_weights_only(input_path, output_path):
    """Load model and save weights only"""
    print("\n[Attempt 4] Extracting weights only...")
    
    try:
        from tensorflow import keras
        
        model = keras.models.load_model(input_path, compile=False)
        print("✓ Model loaded successfully!")
        
        weights_path = output_path.replace('.keras', '_weights.h5').replace('.h5', '_weights.h5')
        model.save_weights(weights_path)
        print(f"✓ Weights saved to: {weights_path}")
        print("Note: This only saves weights. You'll need the architecture to use them.")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert H5 Keras models to .keras format (latest TensorFlow)'
    )
    parser.add_argument(
        'input',
        help='Input H5 model file path'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output model file path (default: model_converted.keras)',
        default='model_converted.keras'
    )
    parser.add_argument(
        '--inspect-only',
        action='store_true',
        help='Only inspect the H5 file without converting'
    )
    parser.add_argument(
        '--format',
        choices=['keras', 'savedmodel', 'weights'],
        default='keras',
        help='Output format (default: keras)'
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
        
        # Check if using Keras 3
        from tensorflow import keras
        keras_version = keras.__version__
        print(f"Keras version: {keras_version}")
        
        if keras_version.startswith('3.'):
            print("Note: Using Keras 3 (may have different behavior)")
        
    except ImportError:
        print("TensorFlow: Not installed")
        sys.exit(1)
    
    # Inspect the file
    inspect_h5_file(args.input)
    
    if args.inspect_only:
        print("\nInspection complete (--inspect-only flag set)")
        return
    
    # Try conversion based on format
    print(f"\nAttempting to convert: {args.input} -> {args.output}")
    print("=" * 60)
    
    if args.format == 'savedmodel':
        methods = [convert_to_savedmodel]
    elif args.format == 'weights':
        methods = [convert_weights_only]
    else:  # keras format
        methods = [
            convert_standard,
            convert_with_safe_mode_off,
        ]
    
    for method in methods:
        if method(args.input, args.output):
            print("\n" + "=" * 60)
            print("✓ CONVERSION SUCCESSFUL!")
            print("=" * 60)
            return
    
    print("\n" + "=" * 60)
    print("✗ CONVERSION FAILED")
    print("=" * 60)
    print("\nSuggestions:")
    print("1. Ensure TensorFlow versions match (creation vs loading)")
    print("2. Try: python -m pip install --upgrade tensorflow")
    print("3. The error above shows the specific issue - share it for help")
    print("4. Try different output formats: --format savedmodel")
    sys.exit(1)


if __name__ == "__main__":
    main()