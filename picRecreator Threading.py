from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
from multiprocessing import Pool
import time


def getRandomPos(x, y):
    return (random.randint(0, x), random.randint(0, y))


def getRandomCircle(xMax, yMax):
    x, y = getRandomPos(xMax, yMax)
    dx = random.randint(0, xMax - x)
    dy = random.randint(0, yMax - y)

    return [(x, y), (x + dx, y + dy)]


def getRandomColor():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b, 255)


def newImage(x, y):
    return Image.new("RGB", (x, y), "White")


def drawRandomCirlce(img):
    draw = ImageDraw.Draw(img)
    draw.ellipse(getRandomCircle(x, y), fill=getRandomColor())


def getFitness(ref, img):
    res = cv2.absdiff(np.array(ref), np.array(img))
    res = res.astype(np.uint8)
    return (np.count_nonzero(res) * 100) / res.size


def func(baseImg):
    BATCH_SIZE = 10
    imgs = [baseImg.copy() for _ in range(BATCH_SIZE)]

    for j in range(1, len(imgs)):
        drawRandomCirlce(imgs[j])

    fit = [getFitness(ref, img) for img in imgs]

    min_ = 100
    winner = -1
    for j in range(len(fit)):
        if fit[j] < min_:
            min_ = fit[j]
            winner = j

    return imgs[winner], fit[winner]


ref = Image.open('reference.jpg')
x, y = ref.size
if __name__ == '__main__':
    EPOCHS = 10
    start_time = time.time()
    base = newImage(x, y)
    for i in range(EPOCHS):
        with Pool(7) as p:
            res = p.map(func, [base for _ in range(100)])
            min_ = 100
            winner = -1
            for j in res:
                img, fitness = j
                if fitness < min_:
                    min_ = fitness
                    winner = img
        print(f'Epoch: {i}, Fitness: {min_}')
        base = winner
        base.save(f'Steps\\Epoch{i}.jpg')

    print("--- %s seconds ---" % (time.time() - start_time))
    plt.imshow(base)
    plt.show()
