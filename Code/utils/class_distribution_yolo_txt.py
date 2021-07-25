import os

label_path = "/media/sanchit/current_working_datasets/UAV/yolov5_images3/train/labels"

class_value_mapping = {}
for i in os.listdir(label_path):
    txt_file = open(os.path.join(label_path, i), "r")
    for j in txt_file.readlines():
        c = int(j.split(" ")[0])
        if c in class_value_mapping.keys():
            class_value_mapping[c] += 1
        else:
            class_value_mapping.update({c: 1})
print(class_value_mapping)