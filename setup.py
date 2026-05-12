from setuptools import setup, find_packages

setup(
    name="TFlow",
    version="0.1.0",
    description="TFlow (Thought Flow): inference-only weight-space inter-agent communication evaluator",
    author="TFlow Authors",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "accelerate>=0.30.0",
        "tqdm>=4.66.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
        "sympy>=1.12",
        "antlr4-python3-runtime==4.11.0",
        "math-verify>=0.5.0",
    ],
)
