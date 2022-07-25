import os
import cv2
import shutil


if __name__ == "__main__":
    # open txt file
    # convert to yolo
    # write new txt file

    fcos_txt_path = "UAV_traffic_pollution_estimation/txt_files/training_txt_final.txt"
    output_folder = "UAV/yolov5_images3/train"

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels"), exist_ok=True)

    fcos_f = open(
        fcos_txt_path,
        "r",
    )
    from tqdm import tqdm

    for idx1, i in tqdm(enumerate(fcos_f.readlines())):
        shutil.copyfile(
            i.split(" ")[0].replace("\n", ""),
            os.path.join(
                output_folder,
                "images",
                str(idx1)
                + "."
                + i.split(" ")[0].split("/")[-1].replace("\n",
                                                         "").split(".")[1],
            ),
        )

        h, w, _ = cv2.imread(i.split(" ")[0].replace("\n", "")).shape
        for idx, bbox in enumerate(i.split(" ")[1:]):
            if idx == 0:
                f = open(
                    os.path.join(
                        output_folder,
                        "labels",
                        str(idx1) + ".txt",
                    ),
                    "w+",
                )
            bbox = bbox.split(",")
            f.write(
                "{} {} {} {} {}\n".format(
                    int(bbox[4]),
                    ((int(bbox[0]) + int(bbox[2])) / 2) / w,
                    ((int(bbox[1]) + int(bbox[3])) / 2) / h,
                    (int(bbox[2]) - int(bbox[0])) / w,
                    (int(bbox[3]) - int(bbox[1])) / h,
                )
            )
