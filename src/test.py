# from model import Model
from processing import process_image
import cv2 as cv


def main():
    image = cv.imread("img\\eqhw.jpg")
    process_image(cv.resize(image, (400, 400)))


if __name__ == "__main__":
    main()
