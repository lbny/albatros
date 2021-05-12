from setuptools import find_packages, setup

from swann import make_tmp_folder

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as input_stream:
    requirements = input_stream.readlines()
    input_stream.close()

make_tmp_folder()

setup(
    name="albatros", # Replace with your own username
    version="0.0.15",
    author="lbny",
    author_email="lb.bony@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lbny/albatros",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires='>=3.6',
)