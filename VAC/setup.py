from setuptools import setup, find_packages

setup(
    name="water_quality_classification",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A water quality classification system",
)
