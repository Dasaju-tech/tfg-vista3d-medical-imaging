import importlib

def test_core_imports():
    packages = [
        "torch",
        "monai",
        "nibabel",
        "SimpleITK",
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "skimage",
    ]

    for package in packages:
        importlib.import_module(package)
