from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import easyocr
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def calculate_ratio(width,height):
    '''
    Calculate aspect ratio for normal use case (w>h) and vertical text (h>w)
    '''
    ratio = width/height
    if ratio<1.0:
        ratio = 1./ratio
    return ratio


def compute_ratio_and_resize(img,width,height,model_height):
    '''
    Calculate ratio and resize correctly for both horizontal text
    and vertical case
    '''
    ratio = width/height
    if ratio<1.0:
        ratio = calculate_ratio(width,height)
        img = cv2.resize(img,(model_height,int(model_height*ratio)), interpolation=Image.ANTIALIAS)
    else:
        img = cv2.resize(img,(int(model_height*ratio),model_height),interpolation=Image.ANTIALIAS)
    return img,ratio


def get_image_list(horizontal_list, free_list, img, model_height = 64, sort_output = True):
    image_list = []
    maximum_y, maximum_x, _ = img.shape

    max_ratio_hori, max_ratio_free = 1,1
    for box in free_list:
        rect = np.array(box, dtype = "float32")
        transformed_img = four_point_transform(img, rect)
        ratio = calculate_ratio(transformed_img.shape[1],transformed_img.shape[0])
        new_width = int(model_height*ratio)
        if new_width == 0:
            pass
        else:
            crop_img,ratio = compute_ratio_and_resize(transformed_img,transformed_img.shape[1],transformed_img.shape[0],model_height)
            image_list.append( (box,crop_img) ) # box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            max_ratio_free = max(ratio, max_ratio_free)


    max_ratio_free = math.ceil(max_ratio_free)

    for box in horizontal_list:
        x_min = max(0,box[0])
        x_max = min(box[1],maximum_x)
        y_min = max(0,box[2])
        y_max = min(box[3],maximum_y)
        crop_img = img[y_min : y_max, x_min:x_max]
        width = x_max - x_min
        height = y_max - y_min
        ratio = calculate_ratio(width,height)
        new_width = int(model_height*ratio)
        if new_width == 0:
            pass
        else:
            crop_img,ratio = compute_ratio_and_resize(crop_img,width,height,model_height)
            image_list.append( ( [[x_min,y_min],[x_max,y_min],[x_max,y_max],[x_min,y_max]] ,crop_img) )
            max_ratio_hori = max(ratio, max_ratio_hori)

    max_ratio_hori = math.ceil(max_ratio_hori)
    max_ratio = max(max_ratio_hori, max_ratio_free)
    max_width = math.ceil(max_ratio)*model_height
    
    if sort_output:
        image_list = sorted(image_list, key=lambda item: item[0][0][1]) # sort by vertical position
    return image_list, max_width


def is_long(img):
    # check img, trocr cannot recognize long sentence, thus need to check if the image potentially have long sentence
    row, col, _ = img.shape
    print(img.shape, (row * 12) < col)
    return (row * 12) < col


def sum_all_space(idx, array):
    sum_ = 0
    for idx_, val in enumerate(array):
        if idx_ < idx:
            continue
        if val == 1:
            sum_ += 1
        elif val == 0:
            return sum_
    return sum_


def get_all_middle_top(array):
    list_idx = []
    for idx in range(len(array)):
        if idx == 0:
            continue
        top = array[idx] > 0
        bottom = array[idx - 1] == 0
        if top & bottom:
            top = array[idx]
            for idx_ in range(50):
                if (idx_ + idx) >= len(array):
                    continue
                idxn = idx_ + idx
                if (array[idxn] == int(top/2)):
                    list_idx.append((idxn, top))
                    break
    power_split = np.asarray([top for _, top in list_idx])
    thr_power = power_split.max() * 0.85
    list_idx = np.asarray([[idx, val] for idx, val in list_idx if val >= thr_power])
    return list_idx


def split_long(img):
    # must split image to smaller image inorder to create shorter sentence
    sum_px = np.log(img.sum(axis=2).sum(axis=0))
    p85 = sum_px.max() + np.log(0.98)
    th_px = (sum_px > p85)
    sum_th = np.asarray([sum_all_space(idx, th_px) for idx in range(th_px.shape[0])])
    split_idx = get_all_middle_top(sum_th)
    ls_img = []
    start_idx = 0

    for idx, val in split_idx:
        crop2 = img[:,start_idx:idx]
        ls_img.append(crop2)
        start_idx = idx
    ls_img.append(img[:,idx:])
    return ls_img


if __name__ == "__main__":
    reader = easyocr.Reader(['en', 'id'])
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

    # load image from the IAM database (actually this model is meant to be used on printed text)
    path = 'image/X51005447859.jpg'
    image = np.asarray(Image.open(path).convert("RGB"))
    img_cv_grey = np.asarray(Image.open(path).convert("L"))
    horizontal_list, free_list = reader.detect(path)
    # get the 1st result from hor & free list as self.detect returns a list of depth 3
    horizontal_list, free_list = horizontal_list[0], free_list[0]
    ls_result = []
    for bbox in horizontal_list:
        h_list = [bbox]
        f_list = []
        image_list, max_width = get_image_list(h_list, f_list, image)
        img = image_list[0][1]
        if is_long(img):
            ls_img = split_long(img)
            generated_text = ""
            for _img in ls_img:
                pixel_values = processor(images=_img, return_tensors="pt").pixel_values
                _generated_ids = model.generate(pixel_values)
                _generated_text = processor.batch_decode(_generated_ids, skip_special_tokens=True)
                generated_text = generated_text + ' ' + _generated_text[0]
            print(bbox, generated_text)

        else:
            pixel_values = processor(images=img, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(bbox, generated_text)
        ls_result.append([bbox, generated_text])
        #break

    for bbox in free_list:
        h_list = []
        f_list = [bbox]
        image_list, max_width = get_image_list(h_list, f_list, image)
        img = image_list[0][1]
        if is_long(img):
            ls_img = split_long(img)
            generated_text = ""
            for _img in ls_img:
                pixel_values = processor(images=_img, return_tensors="pt").pixel_values
                _generated_ids = model.generate(pixel_values)
                _generated_text = processor.batch_decode(_generated_ids, skip_special_tokens=True)
                generated_text = generated_text + _generated_text[0]
            print(bbox, generated_text)

        else:
            pixel_values = processor(images=img, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(bbox, generated_text)
        ls_result.append([bbox, generated_text])
        #break
    print(ls_result)
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = np.asarray(Image.open(path).convert("RGB"))
    for bbox, text in ls_result:
        if len(np.asarray(bbox).shape) == 1:
            image = cv2.rectangle(image, (bbox[0], bbox[2]), (bbox[1], bbox[3]), (0, 255, 0), 2)
            cv2.putText(image, text, (bbox[0], bbox[2]), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
    plt.imshow(image); plt.show()
    image = Image.fromarray(np.uint8(image))
    image.save('image/result.jpg')