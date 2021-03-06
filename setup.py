# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(name='hdc2021_StepNet',
      version='0.0.1',
      description='StepNet for Deblurring',
      url='https://github.com/mschmidt25/hdc2021_StepNet',
      author='Maximilian Schmidt, Alexander Denker, Johannes Leuschner, et. al.',
      author_email='maximilian.schmidt@uni-bremen.de',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'pytorch-lightning>=1.3.8',
          'torch>=1.9.0',
          'torchvision>=0.10.0',
          'dival>=0.6.1',
          'pytesseract>=0.3.8',
          'fuzzywuzzy>=0.18.0'
	   ],
      include_package_data=True,
      zip_safe=False)
