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
import logging
import requests
from .models import model_dict, deputy_files_name


http_headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"}

def download_models(model_name=list(), save_path=str()):
        model_path = os.path.join(save_path, model_name)
        logging.debug("download.download_models.model_path: {}".format(model_path))

        result = requests.get(url=model_dict[model_name], headers=http_headers, timeout=10)

        if result.status_code == 200:
            if os.path.splitext(model_name)[-1] in deputy_files_name:
                with open(model_path, "w") as f:
                    f.write(result.text)
                    logging.info("Saved {} to {}.".format(model_name, model_path))
            else:
                with open(model_path, "wb") as f:
                    f.write(result.content)
                    logging.info("Saved {} to {}.".format(model_name, model_path))