from setuptools import setup


setup(
    name="pyfuzzy",
    version="0.1.0",
    author="Juan D. Velasquez",
    author_email="jdvelasq@unal.edu.co",
    license="MIT",
    url="http://github.com/jdvelasq/pyfuzzy",
    description="Fuzzy Inference Systems",
    long_description="Fuzzy Inference Systems",
    keywords="fuzzy",
    platforms="any",
    provides=["pyfuzzy"],
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    packages=[
        "pyfuzzy",
    ],
    package_dir={"pyfuzzy": "pyfuzzy"},
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
