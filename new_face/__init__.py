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

import os
from .tools.config import root_dir

from .kernel.landmark import FaceLandmark
from .kernel.detection import FaceDetection
from .kernel.alignment import FaceAlignment
from .kernel.lbph import LBPH
from .kernel.openface import OpenFace
from .kernel.lbpcnn import LBPCNN


__author__ = "Overcomer"
__version__ = "0.0.4.5"
__module__ = ["kernel", "tools"]
__classes__ = ["FaceDetection", "FaceLandmark", "FaceAlignment", "LBPH", "OpenFace", "LBPCNN"]


# Build new_face root directory
if not os.path.exists(root_dir):
    os.makedirs(root_dir)