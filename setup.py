import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='brute_curvefit',
    version='1.0.0',
    scripts=['brute_curvefit.py'],
    author="Anal Kumar",
    author_email="analkumar2@gmail.com",
    description="Fits a given function with any number of parameters to a given array of values using brute force as well as scipy optimize. Then returns the parameters.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/analkumar2/brute_curvefit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
)
