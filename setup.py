import setuptools

with open("README.md", "r", encoding="utf8") as readme:
    long_description = readme.read()

setuptools.setup(
    name="new_face",
    version="0.0.1",
    author="Overcomer",
    author_email="michael31703@gmail.com",
    description="Face Recognition Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Michael07220823/new_face.git",
    keywords="Face Recognition",
    install_requires=["opencv-python>=4.4.0.36", "opencv-contrib-python>=4.4.0.36", "tensorflow==2.5.0", "mtcnn", "sklearn", "imutils", "cmake", "dlib", "scikit-image", "new_tools", "pydot"],
    license="MIT License",
    packages=setuptools.find_packages(include=["new_face", "new_face.*"], exclude=["__pycache__"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS"],
)