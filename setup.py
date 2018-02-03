from setuptools import setup
from setuptools import find_packages

from shutil import copyfile, rmtree
import os
import glob

def build_package():
    try:
        rmtree('sklearn_explain')
    except:
        pass
    os.mkdir('sklearn_explain')
    os.mkdir('sklearn_explain/reason_codes')
    for file in glob.glob('reason_codes/*.py'):
        copyfile(file, 'sklearn_explain/' + file)
    copyfile('explainer.py' ,
             'sklearn_explain/explainer.py')
    copyfile('README.md' ,
             'sklearn_explain/README.md')
    copyfile('LICENSE' ,
             'sklearn_explain/LICENSE')
    copyfile('__init__.py',
             'sklearn_explain/__init__.py')


build_package();

setup(name='sklearn_explain',
      version='1.0',
      description='sklearn_explain',
      author='Antoine CARME',
      author_email='antoine.carme@laposte.net',
      url='https://github.com/antoinecarme/sklearn_explain',
      license='BSD 3-clause',
      packages=['sklearn_explain' , 'sklearn_explain/reason_codes'],
      install_requires=[
          'scipy',
          'pandas',
          'sklearn',
      ])

rmtree('sklearn_explain')
