from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    This function reads a requirements file and returns a list of packages.
    '''
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements

setup(
    name='salary_predictor',
    version='0.0.1',
    author='Akinsanya Joel',
    author_email='akinsanyajoel82@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
