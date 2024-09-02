import os

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import fastapi
import copy
import numpy as np
import json
import time
import logging
from PIL import Image
from loguru import logger  # noqa

import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image


class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno + self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'csl': 0, 'all': 0}
        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        time_dict['det'] = elapse
        logger.debug("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            time_dict['cls'] = elapse
            logger.debug("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict['rec'] = elapse
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, 0, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


class Need:
    text_sys = None
    font_path = ''
    drop_score = ''
    draw_img_save_dir = ''


def load_model():
    args = utility.parse_args()
    args.det_model_dir = os.getenv('DET_MODEL', './ch_PP-OCRv3_det_infer')
    args.rec_model_dir = os.getenv('REC_MODEL', './ch_PP-OCRv3_rec_infer')
    logger.info('model={}, {}'.format(args.det_model_dir, args.rec_model_dir))
    args.use_gpu = True
    Need.text_sys = TextSystem(args)
    Need.font_path = args.vis_font_path
    Need.drop_score = args.drop_score
    Need.draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(Need.draw_img_save_dir, exist_ok=True)


def ocr(img_array, download_filename=None):
    _st = time.time()
    start_time = time.time()
    dt_boxes, rec_res, time_dict = Need.text_sys(img_array)
    elapse = time.time() - start_time

    logger.debug("Predict time of %s: %.3fs" % (img_array, elapse))
    for text, score in rec_res:
        logger.debug("{}, {:.3f}".format(text, score))

    res = [{
        "transcription": rec_res[idx][0],
        "points": np.array(dt_boxes[idx]).astype(np.int32).tolist(),
    } for idx in range(len(dt_boxes))]

    if download_filename:
        image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        boxes = dt_boxes
        txts = [rec_res[i][0] for i in range(len(rec_res))]
        scores = [rec_res[i][1] for i in range(len(rec_res))]

        draw_img = draw_ocr_box_txt(
            image,
            boxes,
            txts,
            scores,
            drop_score=Need.drop_score,
            font_path=Need.font_path)
        cv2.imwrite(
            os.path.join(Need.draw_img_save_dir, os.path.basename(download_filename)),
            draw_img[:, :, ::-1])
        logger.debug("The visualized image saved in {}".format(
            os.path.join(Need.draw_img_save_dir, os.path.basename(download_filename))))

    logger.info("The predict total time is {}".format(time.time() - _st))
    return res


if __name__ == "__main__":
    pic_path = './22.png'
    load_model()
    ocr(cv2.imread(pic_path))
