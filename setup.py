import setuptools
from codecs import open
from os import path


from geo_fab import __version__

ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))




setuptools.setup(name='geometric fabrics',
      version=__version__,
      description='Geometric Fabrics',
      author='Julen Urain',
      author_email='julen@robot-learning.de',
      url='https://thecamusean.github.io/',
      packages=['geo_fab'],
      install_requires=requires_list,
     )
