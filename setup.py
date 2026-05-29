from setuptools import setup, find_packages

setup(
    name="uav-swarm-optimization",
    version="1.0.0",
    description="Multi-Objective Bio-Inspired Algorithms for UAV Swarm Collaboration",
    author="Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "plotly>=5.14.0",
        "pymavlink>=2.4.37",
        "mavsdk>=1.4.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.1",
        "h5py>=3.9.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "colorama>=0.4.6",
        "tabulate>=0.9.0",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "uav-sim=main:main",
        ],
    },
)
