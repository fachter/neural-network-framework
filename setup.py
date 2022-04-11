from setuptools import setup, find_packages


VERSION = '0.0.3'
DESCRIPTION = 'Neural Network package'

# Setting up
setup(
    name="netneural",
    version=VERSION,
    author="fachter (Felix)",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=["numpy", "pandas", "pyyaml", "matplotlib", "jupyter", "tqdm", "python-mnist", "seaborn"],
    keywords=['python', 'neural network', 'deep learning', 'nn', 'machine learning', 'neural net'],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
