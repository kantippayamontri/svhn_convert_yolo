import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import os
import shutil
import yaml
import magic
import random
from icecream import ic
import cv2

colors = {
    0: (255, 0, 0),  # Red
    1: (0, 255, 0),  # Green
    2: (0, 0, 255),  # Blue
    3: (255, 255, 0),  # Yellow
    4: (255, 165, 0),  # Orange
    5: (128, 0, 128),  # Purple
    6: (255, 192, 203),  # Pink
    7: (0, 255, 255),  # Cyan
    8: (255, 0, 255),  # Magenta
    9: (0, 255, 0),  # Lime
    10: (0, 128, 128),  # Teal
    11: (75, 0, 130),  # Indigo
    12: (238, 130, 238),  # Violet
    13: (165, 42, 42),  # Brown
    14: (0, 0, 0),  # Black
    15: (255, 255, 255),  # White
    16: (128, 128, 128),  # Gray
    17: (192, 192, 192),  # Silver
    18: (255, 215, 0),  # Gold
    19: (0, 0, 128),  # Navy
}

def get_img_dim(filepath):
    file_info = magic.from_file(filename=filepath)
    # Extract dimensions (might need parsing)
    dimensions_str = file_info.split(',')[1].strip()  # Example: '800 x 600'
    _image_width, _image_height = map(int, dimensions_str.split(' x ')) 
    return _image_width, _image_height

def write_yaml(data, filepath):
    with open(str(filepath), "w") as file:
        yaml.dump(data, file)

def make_data_yaml_dict(nc, names):
    data_yaml_dict = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": nc,
        "names": names,
    }
    return data_yaml_dict

def check_folder_exists(folder_path):
    if os.path.exists(folder_path):
        return True
    return False


def delete_folder_mkdir(folder_path, remove=False):
    if check_folder_exists(folder_path):
        if remove:
            shutil.rmtree(folder_path)
        else:
            print(f"--- This folder exists ---")
            return False
    os.makedirs(folder_path)
    return True


def copy_file(source_file_path, target_file_path):
    shutil.copy2(str(source_file_path), str(target_file_path))
    return


def overwrite_label(txt_file_path, bb):
    file = open(txt_file_path, "w")
    # Write new content to the file
    str_save = ""
    for _bb in bb:
        str_save += f"{int(_bb[0])} {_bb[1]} {_bb[2]} {_bb[3]} {_bb[4]}\n"
    file.write(str_save)
    file.close()

class Box:
    def __init__(self, label, left, top, height, width):
        if int(label) == 10:
            self.label = 0
        else:
            self.label = int(label)
        self.left = int(left)
        self.height = int(height)
        self.width = int(width)
        self.top = int(top)

    def cvtXYXY(self):
        x_min = self.left
        y_min = self.top
        x_max = int(x_min + self.width)
        y_max = int(y_min + self.height)
        return [x_min, y_min, x_max, y_max]

    def cvtYOLO(self):
        x_min, y_min, _, _ = self.cvtXYXY()
        x = x_min + int(self.width / 2)
        y = y_min + int(self.height / 2)
        return [x, y, self.width, self.height]

    def cvtYOLO_n(self, image_width, image_height):
        x, y, w, h = self.cvtYOLO()

        x = x / image_width
        w = w / image_width

        y = y / image_height
        h = h / image_height

        return [x, y, w, h]

    def cvtRoboflowFormat(self, image_height, image_width):
        yolo_n = self.cvtYOLO_n(image_height=image_height, image_width=image_width)
        yolo_n.insert(0, self.label)
        return yolo_n


class DataMat:
    def __init__(self, name=None, folder_path=None, bb=None):
        self.name = name
        self.folder_path = folder_path
        self.file_path = self.folder_path / self.name
        self.__bb = bb
        self.box = []
        if isinstance(self.__bb["height"], list):
            for bb_index in range(len(self.__bb["label"])):
                self.box.append(
                    Box(
                        label=self.__bb["label"][bb_index],
                        left=self.__bb["left"][bb_index],
                        top=self.__bb["top"][bb_index],
                        height=self.__bb["height"][bb_index],
                        width=self.__bb["width"][bb_index],
                    )
                )
        else:
            self.box.append(
                Box(
                    label=self.__bb["label"],
                    left=self.__bb["left"],
                    top=self.__bb["top"],
                    height=self.__bb["height"],
                    width=self.__bb["width"],
                )
            )

def visualize_img_bb(img, bb, with_class=False, format=None, labels=None):
    xyxy_bb = bb

    plt.imshow(img)
    plt.axis("off")  # Turn off axes numbers and ticks

    for xyxy in xyxy_bb:
        ic(xyxy)
        color_index = 0
        top_left = (0, 0)
        bottom_right = (0, 0)
        if with_class:
            color_index = int(xyxy["class"])
            top_left = (xyxy["bb"][0], xyxy["bb"][1])
            bottom_right = (xyxy["bb"][2], xyxy["bb"][3])
        else:
            top_left = (xyxy["bb"][0], xyxy["bb"][1])
            bottom_right = (xyxy["bb"][2], xyxy["bb"][3])

        # create bounding box
        bbox = patches.Rectangle(
            xy=top_left,
            width=bottom_right[0] - top_left[0],
            height=bottom_right[1] - top_left[1],
            linewidth=2,
            edgecolor=np.array(colors[color_index]) / 255.0,
            facecolor="none",
        )

        # add the bounding box rectangle to the current plot
        plt.gca().add_patch(bbox)
        # add text to the bounding box
        if labels != None:
            plt.text(
                top_left[0],
                top_left[1] - 1,
                labels[color_index],
                color=np.array(colors[color_index]) / 255.0,
            )

    plt.show()

def get_filenames_folder(
    source_folder,
):
    return [
        source_folder / file.name
        for file in source_folder.iterdir()
        if file.is_file()
        ]

def change_filename_sample(
    filepath, filename, index, start_index=0, extension=None
):
    if index == start_index:
        if extension == None:
            return filepath
        else:
            return filepath.with_suffix(extension)
    else:
        if extension == None:
            ext = Path(filename).suffix
            filename_with_extension = Path(filename).stem + f"_{index}" + ext
            return filepath.parent / filename_with_extension
        else:
            filename_with_extension = Path(filename).stem + f"_{index}" + extension
            return filepath.parent / filename_with_extension

def match_img_bb_filename(
    img_filenames_list=None, bb_filenames_list=None, source_folder=None
):

    match_img_bb = []
    bb_folder_path = ""

    if source_folder != None:
        bb_folder_path = source_folder / "labels"
    else:
        bb_folder_path = bb_filenames_list[0].parent

    for index, img_filename in enumerate(img_filenames_list):
        # find match bounding box
        filename = Path(img_filename)
        filename = change_filename_sample(
            filepath=filename, filename=None, index=0, extension=".txt"
        ).name
        label_full_path = bb_folder_path / filename

        if label_full_path.is_file():
            match_img_bb.append((img_filename, label_full_path))

    return match_img_bb

def get_filename_bb_folder(img_path=None, bb_path=None, source_folder=None):
    img_filenames = get_filenames_folder(img_path)
    bb_filenames = get_filenames_folder(bb_path)
    match_files = match_img_bb_filename(
        img_filenames_list=img_filenames,
        bb_filenames_list=bb_filenames,
        source_folder=source_folder,
    )
    return match_files

def load_img_cv2(filepath):
    img = cv2.imread(str(filepath))
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except:
        return None

def load_bb(filepath):
    bb = []

    try:
        fp = open(str(filepath), "r")  # read the bounding box
        for c, line in enumerate(fp):
            bb_l = line.split(" ")
            if bb_l[-1] == "\n":
                bb_l = bb_l[: len(bb_l) - 1]
            bb_l = list(float(n) for n in bb_l)
            if len(bb_l) == 5:
                bb.append(bb_l)

        return np.array(bb) if len(bb) > 0 else None
    except:
        return None

def convert_xyxy2yolo(bb):
    cls, x_min, y_min, x_max, y_max = bb[0]
    w = (x_max-x_min)
    h = (y_max-y_min)
    x_c = x_min + (w/2)
    y_c = y_min + (h/2)

    return np.array([[cls, x_c, y_c, w,h]])

def visualize_samples_for_gauge(source_folder, number_of_samples=10, labels=None): # this function for visulize image from roboflow number format to roboflow gauge format

    # TODO: get filenames and bb and labels
    # ic(source_folder)
    img_path = source_folder / 'images'
    bb_path = source_folder / "labels"
    match_filename_bb = get_filename_bb_folder(
        img_path=img_path, bb_path=bb_path, source_folder=source_folder
    )

    # print(f"--- Match File ---")
    # TODO: random images and bb
    number_of_images = len(match_filename_bb)

    random_index_list = []

    if number_of_samples > number_of_images:
        print(
            f"[warning] we can show only {number_of_images} images because number of samples is exceed."
        )
        random_index_list = list(range(number_of_images))
    else:
        while len(random_index_list) < number_of_samples:
            index = random.randint(0, number_of_images - 1)
            if index not in random_index_list:
                random_index_list.append(index)

    # print(f"randon index list : {random_index_list}")
    print(f"number of samples : {number_of_samples}")
    print(f"number of images : {number_of_images}")

    # TODO: visualize image and bb
    for index in random_index_list:
        _img_path = match_filename_bb[index][0]
        _bb_path = match_filename_bb[index][1]
        _img = load_img_cv2(filepath=_img_path)
        _img_h, _img_w = _img.shape[0], _img.shape[1]
        # ic(_img_h, _img_w)
        ic(_img.shape)
        _bb = load_bb(filepath=_bb_path)
        ic(_bb)

        # change from xywh to xyxy format
        for index,_ in enumerate(_bb):
            _bb_x,  _bb_y, _bb_w, _bb_h,  = _bb[index][1:]  
            # for x max, y max 
            _bb[index][1], _bb[index][3] = _bb_x - (_bb_w/2), _bb_x + (_bb_w/2)
            _bb[index][2], _bb[index][4] = _bb_y - (_bb_h/2), _bb_y + (_bb_h/2)
        
        # change xyxy format from range 0-1 to xyxy format from range 0-(image width, image height)
        for index in range(len(_bb)):
            _bb[index][0+1] , _bb[index][2+1] = _bb[index][0+1] * _img_w, _bb[index][2+1] * _img_w # real coor in image for x 
            _bb[index][1+1] , _bb[index][3+1] = _bb[index][1+1] * _img_h, _bb[index][3+1] * _img_h # real cor in image for y
        print(_bb_path, _bb)
        if _img is not None and _bb is not None:
            visualize_img_bb(
                img=_img,
                bb=list({"class":_bb_temp[0], "bb":_bb_temp[1:]} for _bb_temp in _bb),
                with_class=True,
                labels=labels,
            )