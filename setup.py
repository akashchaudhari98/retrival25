from setuptools import setup, find_packages

setup(
    name='retrival25',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here.
        # e.g., 'requests', 'numpy'
        'numpy',
        'pandas'
    ],
    author='Akash Chaudhari',
    author_email='akashchaudhari726@gmail.com',
    description="One place stop for BM25 and it's variants",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/akashchaudhari98/retrival25',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
