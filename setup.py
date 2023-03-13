from setuptools import setup, find_packages

setup(
    name='SHARC',
    version='0.1.0',
    author="Marten Lourens",
    packages=find_packages(include=['SHARC', 'SHARC.*']),
    install_requires=[
        'numpy>=1.20.3',
        'matplotlib>=3.4.3',
        'scikit-learn>=1.1.0',
        'tensorflow>=2.8.0',
        'scipy>=1.7.1',
        'pandas>=1.3.4',
        'seaborn>=0.11.2',
        'pySDR>=0.1.0'
    ]
    )