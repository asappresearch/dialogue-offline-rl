from io import open
from setuptools import find_packages, setup


def read_requirements_file(filename):
    with open(filename) as f:
        return [line.strip() for line in f]

setup(
    name="dialogue_offline_rl",
    version="0.0.1",
    author="Paloma Sodhi",
    author_email="psodhi@asapp.com",
    description="Offline RL for Dialogue Response Generation",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="offline rl, dialogue, generative models",
    license="MIT",
    url="",
    package_dir={'': 'src'},
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=read_requirements_file("requirements.txt"),
    entry_points={},
    include_package_data=True,
    python_requires=">=3.6",
    tests_require=["pytest"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
