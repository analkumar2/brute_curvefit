import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="brute_curvefit", # Replace with your own username
    version="0.0.3",
    author="Anal Kumar",
    author_email="analkumar2@gmail.com",
    description="Curve fitting using both brute force and scipy.optimize",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/analkumar2/brute_force",
    packages = [ "brute_curvefit" ],
    package_dir = { "brute_curvefit" : 'brute_curvefit' },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
