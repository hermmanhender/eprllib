from distutils.core import setup
import sys
import os

# Add the src directory to the path so we can import the version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from eprllib.version import __version__
setup(
  name = 'eprllib',         # Nombre de la carpeta (MiLibreria)
  packages = ['eprllib'],   # El mismo que en "name"

  version = __version__,      # Versión importada desde version.py

  license='MIT',        # Escoje una licencia: https://help.github.com/articles/licensing-a-repository
  description = "program used to investigate the control of natural ventilation in homes based on a DRL model. The program uses the EnergyPlus Python API and Ray's Tune and RLlib libraries.",   # Descripcion corta de mi libreria
  author = 'Germán Rodolfo Henderson',                   # Tu nombre
  author_email = 'hendger.07@gmail.com',      # Tu correo electrónico
  url = 'https://github.com/hermmanhender/eprllib',   # Tu github o tu sitio web

  download_url = f'https://github.com/hermmanhender/eprllib/dist/eprllib-{__version__}.tar.gz',    # URL de descarga con versión dinámica

  keywords = ['RLlib', 'DRL', 'EnergyPlus'],   # Palabras que describan tu librería
  install_requires=[            # Los paquetes que usas en tu librería.
          "ray[all] >=2.20.0",
          "gymnasium >=0.28.1",
          "torch >=2.5.1",
          "shap >=0.46.0",
          "matplotlib >=3.8.0"
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Escoje entre "3 - Alpha", "4 - Beta" or "5 - Production/Stable" como el estado actual de tu librería
    'Intended Audience :: Developers',      # Define tu audiencia
    'Topic :: Building Control :: Reinforcement Learning', # Define tu tema
    'License :: OSI Approved :: MIT License',   # Tu licencia
    'Programming Language :: Python :: 3',
  ],
)
