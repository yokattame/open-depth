from setuptools import setup, find_packages
import os
import os.path as osp

#cur_path = osp.abspath('.')
#os.chdir('depth/networks/correlation_package')
#os.system('./make.sh')
#os.chdir(cur_path)

setup(name='open-depth',
      version='0.0.1',
      author='yokatta.me',
      author_email='yokatta.me@gmail.com',
      install_requires=['numpy', 'torch', 'torchvision',],
      packages=find_packages(),
      )

