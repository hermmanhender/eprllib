from distutils.core import setup
setup(
  name = 'eprllib',         # Nombre de la carpeta (MiLibreria)
  packages = ['eprllib'],   # El mismo que en "name"
  version = '1.0',      # Recuerda incrementar la versión con cada cambio
  license='MIT',        # Escoje una licencia: https://help.github.com/articles/licensing-a-repository
  description = "program used to investigate the control of natural ventilation in homes based on a DRL model. The program uses the EnergyPlus Python API and Ray's Tune and RLlib libraries.",   # Descripcion corta de mi libreria
  author = 'Germán Rodolfo Henderson',                   # Tu nombre
  author_email = 'hendger.07@gmail.com',      # Tu correo electrónico
  url = 'https://github.com/hermmanhender/natural_ventilation_EP_RLlib',   # Tu github o tu sitio web
  download_url = 'https://github.com/hermmanhender/natural_ventilation_EP_RLlib/archive/refs/tags/v1.0.2.tar.gz',    # Ahora hablamos de esto
  keywords = ['RLlib', 'DRL', 'EnergyPlus'],   # Palabras que describan tu librería
  install_requires=[            # Los paquetes que usas en tu librería.
          'tensorflow',
          'torch',
          'ray[all]',
          'bayesian-optimization',
          'gymnasium'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Escoje entre "3 - Alpha", "4 - Beta" or "5 - Production/Stable" como el estado actual de tu librería
    'Intended Audience :: Developers',      # Define tu audiencia
    'Topic :: Building Control :: Reinforcement Learning', # Define tu tema
    'License :: OSI Approved :: MIT License',   # Tu licencia
    'Programming Language :: Python :: 3.10',
  ],
)
