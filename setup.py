from setuptools import setup, find_packages

setup(
    name="decypher",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "networkx",
        "statsmodels",
        "scipy",
        "scikit-learn",
        "plotly"
    ],
    author="Evan Peikon",
    description="Decypher - Time series causal analysis tool with mediation analysis",
    url="https://github.com/evanpeikon/Decypher",
)
