from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='bangkits',
      version='0.1',
      description='Indonesian Text Generator using Synonim',
      long_description=readme(),
      url='https://github.com/l1th1um/bangkits',
      author='Andri Fachrur Rozie',
      author_email='rozie.andri@gmail.com',
      license='MIT',
      packages=find_packages(),
      package_data={'bangkits': ['bangkits/data/*.*']},
      include_package_data = True,
      install_requires=[
          'numpy==1.13.3',
          'pandas==0.21.0',
          'python-dateutil==2.6.1',
          'pytz==2017.3',
          'scikit-learn==0.19.0',
          'scipy==1.0.0',
          'six==1.11.0',
          'sklearn==0.0'
      ],
      zip_safe=False)