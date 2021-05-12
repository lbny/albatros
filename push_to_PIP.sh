#!/usr/bin/sh
source /home/lbony/venv_py_main/bin/activate
cd $HOME/commonlit/
rm -rf ./dist/*
python3 setup.py sdist bdist_wheel
twine upload --repository testpypi dist/*