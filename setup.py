from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
        name='augment-nd',
        version='0.1.1',
        description='Simple elastic augmentation for ND arrays.',
        url='https://github.com/funkey/augment',
        author='Jan Funke',
        author_email='jfunke@iri.upc.edu',
        license='MIT',
        packages=['augment'],
        long_description=long_description,
        long_description_content_type='text/markdown',
        install_requires=['h5py', 'numpy', 'scipy'],
        classifiers=[
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ]
)

