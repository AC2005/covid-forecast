from setuptools import setup, find_packages
import model

setup(
    name="covid_learning",
    version=model.__version__,
    author="Andrew Chang",
    author_email="changandrew777@gmail.com",
    description="Covid-19 Prediction",
    packages=["covid"],
    include_package_data=True,
    install_requires=["matplot",
                      "numpy",
                      "pandas",
                      "pytest",
                      "pytest-cov",
                      "tensorflow",
                      "tensorflow_probability",
                      ],
    setup_requires=[
        'pytest-runner'
    ],
    tests_require=[
        'pytest',
        'pytest-cov'
    ]
)