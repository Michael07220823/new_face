"""
MIT License

Copyright (c) 2021 Overcomer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import setuptools

with open("README.pypi.md", "r", encoding="utf8") as readme:
    long_description = readme.read()

setuptools.setup(
    name="new_face",
    version="0.0.4.5",
    author="Overcomer",
    author_email="michael31703@gmail.com",
    description="Face Recognition Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Michael07220823/new_face.git",
    keywords="Face Recognition",
    install_requires=[
        "tensorflow>=2.0.0",
        "scikit-learn==1.0.1",
        "sklearn",
        "mtcnn",
        "imutils",
        "cmake",
        "dlib",
        "scikit-image",
        "matplotlib",
        "opencv-python",
        "opencv-contrib-python",
        "pydot",
        "gdown",
        "new_tools",
        "new_timer",
    ],
    license="MIT License",
    packages=setuptools.find_packages(include=["new_face", "new_face.*"], exclude=["__pycache__"]),
    classifiers=[
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License"
        ]
)