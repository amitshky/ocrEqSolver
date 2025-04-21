import os
from os import listdir
import cv2 as cv


def count_dataset(path: str = "dataset/"):
    map: dict = dict()
    for files in listdir(path):  # type: str
        symbol, _ = files.split("-")
        if symbol in map:
            map[symbol] += 1
            continue

        map[symbol] = 1

    print(map)


def process_images(dir_path: str, processed_dir: str):
    for file in listdir(dir_path):
        if os.path.isdir(file):
            continue

        [filename, extension] = file.split(".")
        if extension != "jpg" and extension != "png":
            continue

        # add / at the end of path if its not there
        dir_path = dir_path if dir_path[-1] == '/' or dir_path[-1] == '\\' \
            else dir_path + '/'
        processed_dir = processed_dir if processed_dir[-1] == '/' \
            or processed_dir[-1] == '\\' \
            else processed_dir + '/'

        img_path = dir_path + file

        image = cv.imread(img_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.bitwise_not(image)
        _, image = cv.threshold(
            image, 128, 255, cv.THRESH_BINARY + cv.THRESH_BINARY)
        image = image.reshape(28, 28, 1)

        split = file.split("-")

        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

        folder = processed_dir + split[0] + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        cv.imwrite(folder + filename + ".jpg", image[:, :, 0])


# organizes images into folders according to their label
def organize_dir(dir_path: str = "processed/"):
    for file in listdir(dir_path):
        img_path = dir_path + file
        split = file.split("-")

        folder = dir_path + split[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        os.rename(img_path, folder + '/' + file)


def main():
    # organize_dir()
    process_images("dataset/", "processed/")


if __name__ == "__main__":
    main()
