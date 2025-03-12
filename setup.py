from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of req
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='mlproject1',
    version='0.0.1',
    author='Razim',
    author_email='razim.manz@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    entry_points={
        "console_scripts": [
            "train_pipeline=train_pipeline:train_pipeline",
            "predict_pipeline=predict_pipeline:predict"
        ]
    },
    description="A machine learning pipeline project with Flask web app integration"
)