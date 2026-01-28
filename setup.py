from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neural-cryptanalyst",
    version="0.2.0",
    author="Ahmed Taha",
    author_email="ataeha1@jh.edu",
    description="Machine learning toolkit for side-channel analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahmedtaha100/Neural-Cryptanalyst",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Security :: Cryptography",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "tensorflow>=2.8.0,<2.14.0",
        "matplotlib>=3.4.0",
        "h5py>=3.1.0",
        "joblib>=1.0.0",
        "tqdm>=4.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "pytest-cov>=2.12.0"],
        "gpu": ["GPUtil>=1.4.0"],
        "viz": ["seaborn>=0.11.0"],
        "dtw": ["dtaidistance>=2.0.0"],
        "all": [
            "pytest>=6.0",
            "pytest-cov>=2.12.0",
            "GPUtil>=1.4.0",
            "seaborn>=0.11.0",
            "dtaidistance>=2.0.0",
        ],
    },
)
