
# PaddleOCR-API
<p align="center">
<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/stargazers"><img src="https://img.shields.io/github/stars/m986883511/PaddleOCR-API?color=ccf"></a>
</p>

## Introduction

PaddleOCR-API aims to provide an out-of-the-box ocr recognition interface, the ocr algorithm comes from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).

## Docker Build
```shell
docker build -f Dockerfile-env.yml -t m986883511/paddleocr:env .
docker build -f Dockerfile-api.yml -t m986883511/paddleocr:api .
```

## Docker Use
```shell
docker run --rm -it -p 8000:8000 -e DET_MODEL=/build/ch_PP-OCRv3_det_infer -e REC_MODEL=/build/ch_PP-OCRv3_rec_infer --gpus all --name PaddleOCR-API m986883511/paddleocr:api
```

## Browser
When the docker run, open [localhost:8000](http://localhost:8000) you can see the interface.
<div align="center">
    <img src="./doc/paddleocr-api-view.png" width="800">
</div>


<a name="LICENSE"></a>
## 📄 License
This project is released under <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>
