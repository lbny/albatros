# Example Package

This is a simple example package. You can use
[Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
to write your content.

This package is a quickstart example for setting up a
Python package development environment.

A detailed package detailing all elements of a package is available there:
https://github.com/pypa/sampleproject
However documentation is extensive and not suited to quickstart a new project.
Dev guidelines:
https://packaging.python.org/guides/distributing-packages-using-setuptools/#semantic-versioning-preferred
Mainly :
- versioning .devN, .aN, .bN
when looking at X.Y.Z versioning : X is major, meaning incompatible API changes, Y is minor for new functionality but same backward compatible API, Z is for bug fixes
- workin dev mode using
    ```
    pip install -e . #from root of package folder
    ```
    You can edit your package while using it in test file. It is even possible to have multiple projects in editable mode if you're working on project A, using B as a dependency you would create a requirement file :
    ```
    -e /path/to/project/bar
    -e .
    ```
- universal deps are meant to run on python 2 and 3 (you can put this in setup.cfg, universal=1)
- twine can be configured with 
- MANIFEST.in is for auxiliary files to include in the .tar.gz including where to put them in the final package




Requirements:
    - setuptools
    - wheels
    - (opt) sphinx, gcc

- Create minimal structure : 
    

    ./<package_name>/
        ./__init__.py
    ./setup.py
    ./LICENSE
    ./README.md

    use the content in setup.py, LICENSE and README.md
    from this package folder


- (optional) Create sphinx folder for documentation
    ```
    make ./doc
    cd ./doc
    sphinx-quickstart
    ```
    use index.rst to document
    ```
    make html
    ```

- Build first wheel
    ```
    ($ python3 -m pip install --user --upgrade setuptools wheel)
    $ python3 setup.py sdist bdist_wheel
    ```

- (optional) Use a keyring to store username and password for testPyPI
    ```
    $ pip install keyring, keyring.cryptfile
    $ keyring set https://test.pypi.org/legacy/ username
    ```
    set password for your testPyPI account
    set password for the keyring (which you'll have to enter for
    each call to twine)

- Upload distribution files to TestPyPI
    ```
    ($ python3 -m pip install --user --upgrade twine)
    $ twine upload --repository testpypi dist/*
    ```
    Enter your username & password

- Install your package
    ```
    $ pip install --index-url https://test.pypi.org/simple/ brahma
    ```

    