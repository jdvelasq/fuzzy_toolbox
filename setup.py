from setuptools import setup


setup(
    name="fuzzy-toolbox",
    version="0.1.0",
    author="Juan D. Velasquez",
    author_email="jdvelasq@unal.edu.co",
    license="MIT",
    url="http://github.com/jdvelasq/fuzzy-toolbox",
    description="Fuzzy Toolbox",
    long_description="Fuzzy Inference Systems Toolbox",
    keywords="fuzzy",
    platforms="any",
    provides=["fuzzy_toolbox"],
    install_requires=[
        "numpy",
        "matplotlib",
        "progressbar2",
    ],
    packages=[
        "pyfuzzy",
    ],
    package_dir={"fuzzy_toolbox": "pyfuzzy_toolbox"},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Engineering",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
