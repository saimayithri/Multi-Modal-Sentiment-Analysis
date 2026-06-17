"""
Dataset Download & Format Helper for CMU-MOSI.

Downloads data from Google Drive and auto-detects the format.
Works with both .pkl and your existing unaligned_50.pkl files.

Usage (Colab):
    !python run/download_data.py
    
Usage (local):
    python run/download_data.py --output_dir data
"""

import os
import sys
import pickle
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1uEK737LXB9jAlf9kyqRs6B9N6cDncodq"


def download_from_drive(output_dir='data'):
    """Download the CMU-MOSI dataset folder from Google Drive."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we already have data
    existing = [f for f in os.listdir(output_dir) if f.endswith(('.pkl', '.hdf5'))]
    if existing:
        print(f"Data files already exist in {output_dir}/:")
        for f in existing:
            size = os.path.getsize(os.path.join(output_dir, f)) / 1e6
            print(f"  {f}: {size:.1f} MB")
        return
    
    print(f"Downloading CMU-MOSI from Google Drive...")
    print(f"Source: {DRIVE_FOLDER_URL}")
    
    try:
        import gdown
        gdown.download_folder(DRIVE_FOLDER_URL, output=output_dir, quiet=False)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\ngdown folder download failed: {e}")
        print("Trying individual file download...")
        # Fallback: try downloading mosi_data.pkl directly
        print("Please download manually from the Drive link and place in data/ folder.")
        return
    
    # List downloaded files
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            print(f"  {f}: {os.path.getsize(fpath) / 1e6:.1f} MB")


def inspect_pkl(filepath):
    """Inspect a pickle file's structure and print summary."""
    print(f"\n{'='*60}")
    print(f"  Inspecting: {filepath}")
    print(f"  Size: {os.path.getsize(filepath) / 1e6:.1f} MB")
    print(f"{'='*60}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        print(f"\nTop-level keys: {list(data.keys())}")
        
        for split in ['train', 'valid', 'test']:
            if split in data:
                print(f"\n--- {split.upper()} ---")
                if isinstance(data[split], dict):
                    for key, val in data[split].items():
                        if isinstance(val, np.ndarray):
                            print(f"  {key:25s}: shape={str(val.shape):20s} dtype={val.dtype}")
                        elif isinstance(val, (list, tuple)):
                            print(f"  {key:25s}: len={len(val)}")
                        else:
                            print(f"  {key:25s}: type={type(val).__name__}")
                else:
                    print(f"  Type: {type(data[split])}")
    else:
        print(f"Data type: {type(data)}")
    
    return data


def check_compatibility(filepath):
    """Check if a pkl file is compatible with the existing CMUDataset loader.
    
    The loader expects:
      data[split]['text']     -> numpy array [N, SeqLen, Dim]
      data[split]['audio']    -> numpy array [N, SeqLen, Dim]
      data[split]['vision']   -> numpy array [N, SeqLen, Dim]
      data[split]['regression_labels'] -> numpy array [N, 1] or [N]
    """
    print(f"\nChecking compatibility with CMUDataset loader...")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    required_keys = ['text', 'audio', 'vision', 'regression_labels']
    issues = []
    
    for split in ['train', 'valid', 'test']:
        if split not in data:
            issues.append(f"Missing split: '{split}'")
            continue
        
        for key in required_keys:
            if key not in data[split]:
                issues.append(f"Missing key '{key}' in split '{split}'")
            elif isinstance(data[split][key], np.ndarray):
                shape = data[split][key].shape
                if key in ['text', 'audio', 'vision'] and len(shape) != 3:
                    issues.append(f"'{key}' in '{split}' has {len(shape)}D shape {shape}, expected 3D [N, SeqLen, Dim]")
            else:
                issues.append(f"'{key}' in '{split}' is {type(data[split][key]).__name__}, expected ndarray")
    
    if issues:
        print("  ⚠️ Compatibility issues found:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        dims = {
            'text': data['train']['text'].shape[2],
            'audio': data['train']['audio'].shape[2],
            'vision': data['train']['vision'].shape[2],
        }
        n_train = len(data['train']['text'])
        n_valid = len(data['valid']['text'])
        n_test = len(data['test']['text'])
        
        print("  ✅ Fully compatible! Summary:")
        print(f"     Samples: train={n_train}, valid={n_valid}, test={n_test}")
        print(f"     Text dim:   {dims['text']}")
        print(f"     Audio dim:  {dims['audio']}")
        print(f"     Vision dim: {dims['vision']}")
        print(f"\n  Use with: --data_path {filepath}")
        return True


def find_best_data_file(data_dir='data'):
    """Find the best data file to use from available options."""
    candidates = []
    
    for f in os.listdir(data_dir):
        fpath = os.path.join(data_dir, f)
        if f.endswith('.pkl') and os.path.isfile(fpath):
            candidates.append(fpath)
    
    if not candidates:
        print("No .pkl files found in data/ directory.")
        print("Run: python run/download_data.py to download the dataset.")
        return None
    
    print(f"\nFound {len(candidates)} data file(s):")
    compatible = []
    for fpath in candidates:
        print(f"\n  Checking: {os.path.basename(fpath)}")
        if check_compatibility(fpath):
            compatible.append(fpath)
    
    if compatible:
        # Prefer unaligned > data > raw
        preferred_order = ['unaligned', 'data', 'raw']
        for pref in preferred_order:
            for fpath in compatible:
                if pref in os.path.basename(fpath).lower():
                    print(f"\n✅ Recommended: {fpath}")
                    return fpath
        print(f"\n✅ Using: {compatible[0]}")
        return compatible[0]
    else:
        print("\n❌ No compatible data files found.")
        return None


def main():
    parser = argparse.ArgumentParser(description='Download & inspect CMU-MOSI data')
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--inspect', type=str, default=None,
                        help='Path to a specific .pkl file to inspect')
    parser.add_argument('--download', action='store_true', default=True,
                        help='Download from Google Drive')
    args = parser.parse_args()
    
    if args.inspect:
        inspect_pkl(args.inspect)
        check_compatibility(args.inspect)
    else:
        download_from_drive(args.output_dir)
        print("\n" + "="*60)
        print("  COMPATIBILITY CHECK")
        print("="*60)
        best = find_best_data_file(args.output_dir)
        if best:
            print(f"\n🚀 To train, use:")
            print(f"   python run/train.py --stage 1 --data_path {best}")


if __name__ == '__main__':
    main()
