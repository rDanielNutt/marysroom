from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='marysroom',
    version='0.0.1',
    author='Robert Daniel Nutt',
    author_email='rdnutt3@gmail.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rDanielNutt/MarysRoom',
    license='MIT',
    packages=['marysroom'],
    install_requires=[
        'pandas',
        'numpy',
        'plotly',
    ]
)
