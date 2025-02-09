from setuptools import find_packages,setup
from typing import List


hyphon = '-e .'
def get_requirments(file_path:str)->List[str]:

    requirments = []
    with open(file_path) as file_obj:
        requirments = file_obj.readlines()
        requirments = [req.replace('\n',' ') for req in requirments]
    
    if hyphon in requirments:
        requirments.remove(hyphon)

    return requirments

    


setup(
    name= 'Heart_Diseases_Classification',
    author= 'Fadhil Hussain',
    author_email= 'ibnhussainkv@gmail.com',
    version= '1.0.0',
    description= 'Ten year CHD diseases problem',
    long_description= open('README.md').read(),
    url= '',
    packages= find_packages(),
    python_requires = '>=3.8',
    install_requires= get_requirments('requirements.txt'),
)