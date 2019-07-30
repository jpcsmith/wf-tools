"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from os import path
from pathlib import Path
from setuptools import setup, find_packages

DEPENDENCIES = [
    'click',
    'mypy-extensions',
    'pandas',
    'scapy',
    'selenium',
    'typing-extensions',
]


setup(
    name='lab',  # Required
    description='QUIC traffic analysis experiment sources',  # Optional
    # url='https://github.com/pypa/sampleproject',  # Optional

    # Optional long description in README.md
    long_description=Path(
        path.join(path.abspath(path.dirname(__file__)), 'README.md')
    ).read_text(),

    # Automatically extract version information from git tags
    use_scm_version={'root': '../'},
    setup_requires=['setuptools_scm'],

    # This should be your name or the name of the organization which owns the
    # project.
    author='Jean-Pierre Smith',  # Optional

    # This should be a valid email address corresponding to the author listed
    # above.
    author_email='rougesprit@gmail.com',  # Optional

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required

    install_requires=DEPENDENCIES,
    extras_require={  # Optional
        'dev': ['flake8', 'pylint', 'mypy'],
        'test': ['pytest'],
    }
)
