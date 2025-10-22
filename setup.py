"""
Setup script to install SliceBench and verify installation.
"""

import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install required packages."""
    print("Installing SliceBench dependencies...")
    print("This may take a few minutes...\n")
    
    requirements = Path(__file__).parent / "requirements.txt"
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements)
        ])
        print("\n✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Installation failed: {e}")
        return False


def verify_installation():
    """Verify that all components are working."""
    print("\nVerifying installation...")
    
    try:
        subprocess.check_call([sys.executable, "test_slicebench.py"])
        print("\n✓ SliceBench is ready to use!")
        return True
    except subprocess.CalledProcessError:
        print("\n✗ Some tests failed. Please check the output above.")
        return False


def main():
    print("="*60)
    print("SliceBench Setup")
    print("="*60)
    print()
    
    response = input("Install dependencies? (y/n): ").strip().lower()
    
    if response == 'y':
        if install_dependencies():
            verify_installation()
    else:
        print("Skipping installation. Run 'pip install -r requirements.txt' manually.")
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Generate sample dataset: python create_sample_dataset.py")
    print("2. Run quick test: python example_usage.py")
    print("3. Full evaluation: python run_evaluation.py")
    print("4. See README.md for more details")


if __name__ == "__main__":
    main()
