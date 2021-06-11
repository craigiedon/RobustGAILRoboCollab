from setuptools import setup, find_packages

setup(name='gym_reach',
      version='0.0.1',
      install_requires=['gym'],
      packages=['gym_reach', 'gym_reach/envs', 'gym_reach/scenes'],
      include_package_data=True,
      package_data={"": ["*.ttt"]}
)
