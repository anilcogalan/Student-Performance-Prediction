from setuptools import setup, find_packages
from typing import List


HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str)->list[str]:
    """
    Get requirements from the file.

    """

    with open(file_path, "r") as file:
        requirements = file.readlines()
        requirements = [line.replace("\n","")for line in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name='student_performance',
    version='0.1',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    author='Anıl Çoğalan',
    author_email='anilcogalan@outlook.com',
    description='Student Performance Prediction Package',
    url='https://github.com/anilcogalan/Student-Performance-Prediction.git',
)

