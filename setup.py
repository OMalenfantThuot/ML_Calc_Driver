from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy>=1.20,<1.24",
    "torch>=2.0.1",
    "schnetpack==1.0.1",
    "ase>=3.22.0",
]

setup(
    name="mlcalcdriver",
    version="1.4.2",
    author="Olivier Malenfant-Thuot",
    author_email="malenfantthuotolivier@gmail.com",
    description="A package to drive atomic calculations using machine learned models.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/OMalenfantThuot/ML_Calc_Driver",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10",
    ],
)
