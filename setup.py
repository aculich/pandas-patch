from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pandas-patch',
      version= "0.1",
      description='A quick monkey patch to add some methods to the pandas DataFrame class',
      long_description=readme(),
      author='Eric Fourrier',
      author_email='ericfourrier0@gmail.com',
      license = 'MIT',
      url='https://github.com/ericfourrier/pandas-patch',
      packages=['pandas_patch'],
      test_suite = 'test',
      keywords=['monkey','patch', 'pandas'],
      zip_safe=False,
      install_requires=[
          'numpy>=1.7.0',
          'pandas>=0.15.0']
)