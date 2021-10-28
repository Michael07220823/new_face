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

deputy_files_name = [".xml", ".prototxt"]
model_dict = {
    # Haarcascade models.
    "haarcascade_eye.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml",
    "haarcascade_smile.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml",
    "haarcascade_frontalface_default.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",

    # Face detection caffe models.
    "deploy.prototxt": "https://docs.google.com/uc?export=download&id=1LKaeEwj0pUx1-LeKqSBV7gzJ_hat7uu6",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://docs.google.com/uc?export=download&id=1kEV-tp681FraE1lb1RpaqB-yMFLsLVyU",

    # Face landmark models.
    "shape_predictor_5_face_landmarks.dat": "https://docs.google.com/uc?export=download&id=1gc4bOvx4IeR-K3EE1LEW4AYjlymx4R3-",
    "shape_predictor_68_face_landmarks.dat": "https://docs.google.com/uc?export=download&id=1wt8wottsmQCmn46Zk6AoatEIHjFMF3qC",
    
    # Face Recognition openface nn4.small2.v1.t7 model.
    "nn4.small2.v1.t7": "https://docs.google.com/uc?export=download&id=17KpYlFbmYan6FYXPAkV5W9pWK5DC93at"
}