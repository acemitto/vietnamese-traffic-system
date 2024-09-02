import os
import base64
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Depends, UploadFile, File
from starlette.responses import RedirectResponse, FileResponse
from pydantic import BaseModel
from loguru import logger  # noqa

from ocr import load_model, ocr

__dir__ = os.path.dirname(__file__)

app = FastAPI(
    title="PaddleOCR-API",
    # dependencies=[Depends(load_model)],
    description="Paddle OCR API"
)


class OcrBase64Body(BaseModel):
    filename: Optional[str]
    base64: str


class Transfer:

    @staticmethod
    def filepath_to_array(file_content):
         return cv2.imread(file_content)

    @staticmethod
    def base64_to_array(base64_img):
        img = base64.b64decode(base64_img)
        image_data = np.frombuffer(img, np.uint8)
        return cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    @staticmethod
    def bytes_to_array(content):
        image_data = np.frombuffer(content, np.uint8)
        return cv2.imdecode(image_data, cv2.IMREAD_COLOR)


@app.get("/", include_in_schema=False)
async def root(request: Request):
    docs_url = "{}docs".format(request.url)
    return RedirectResponse(url=docs_url)


@app.post("/ocr/dict")
async def ocr_dict(file: UploadFile = File(...)):
    data = await file.read()
    with open(file.filename, 'wb') as f:
        f.write(data)
    array = Transfer.bytes_to_array(data)
    return ocr(array)


@app.post("/ocr/file")
async def ocr_file(file: UploadFile = File(...)):
    filename = "inference_results/{}".format(file.filename)
    data = await file.read()
    with open(file.filename, 'wb') as f:
        f.write(data)
    array = Transfer.bytes_to_array(data)
    ocr(array, download_filename=file.filename)
    return FileResponse(
            filename,  # 这里的文件名是你要发送的文件名
            filename=file.filename  # 这里的文件名是你要给用户展示的下载的文件名，比如我这里叫lol.exe
        )


@app.post("/ocr/base64", summary="对图片的base64编码值进行OCR识别(不要带data:image/png;base64前缀)")
def ocr_base64(ocr_base64_body: OcrBase64Body):
    array = Transfer.base64_to_array(ocr_base64_body)
    return ocr(array)


if __name__ == '__main__':
    LOG_PATH = os.path.join(__dir__, 'paddle_ocr_api.log')
    logger.add(LOG_PATH, rotation="1024KB")
    host, port = '0.0.0.0', 8000
    logger.info('bind on {}:{}, log_path is {}'.format(host, port, LOG_PATH))
    load_model()
    uvicorn.run(app="main:app", host=host, port=port)
