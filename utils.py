import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import os
import shutil
import yaml

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


def visualize_img_bb(img, bb, with_class=False, format=None, labels=None):
    xyxy_bb = bb

    plt.imshow(img)
    plt.axis("off")  # Turn off axes numbers and ticks

    for xyxy in xyxy_bb:
        # ic(xyxy)
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
