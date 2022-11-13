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
        'keras==2.10.0',
        'Keras-Preprocessing==1.1.2',        
        'matplotlib==3.5.3',        
        'nltk==3.7',
        'numpy==1.21.6',        
        'pandas==1.1.5',        
        'requests==2.28.1',        
        'scikit-learn==1.0.2',
        'scipy==1.7.3',                
        'tensorflow==2.10.0',        
        'urllib3==1.26.12',
        'xlrd==2.0.1'        
      ],
      zip_safe=False)