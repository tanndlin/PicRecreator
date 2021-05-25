import time
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np


def getRandomPos(x, y):
    return (random.randint(0, x), random.randint(0, y))


def getRandomCircle(xMax, yMax):
    x, y = getRandomPos(xMax, yMax)
    dx = random.randint(0, xMax - x)
    dy = random.randint(0, yMax - y)

    return [(x, y), (x + dx, y + dy)]


def getRandomTriangle(xMax, yMax):
    return[getRandomPos(xMax, yMax) for _ in range(3)]


def getRandomColor(colors):
    return random.choice(colors)


def newImage(x, y):
    return Image.new("RGB", (x, y), "White")


def drawRandomCircle(img, colors):
    draw = ImageDraw.Draw(img)
    draw.ellipse(getRandomCircle(x, y), fill=getRandomColor(colors))


def drawRandomTriangle(img, colors):
    draw = ImageDraw.Draw(img)
    draw.polygon(getRandomTriangle(x, y), fill=getRandomColor(colors))


def getFitness(ref, img):
    res = cv2.absdiff(np.array(ref), np.array(img))
    res = res.astype(np.uint8)
    return (np.count_nonzero(res) * 100) / res.size


def getColorPallette(img):
    pixels = np.array(img)
    colors = set()
    for i in pixels:
        for j in i:
            col = (j[0], j[1], j[2], 255)
            if col not in colors:
                colors.add(col)
    return [i for i in colors]  # Convert to list


ref = Image.open('reference.jpg')
x, y = ref.size
colors = getColorPallette(ref)

EPOCHS = 10000
BATCH_SIZE = 100
base = newImage(x, y)
winner = -1
last = -1

start_time = time.time()
for i in range(EPOCHS):
    images = [base.copy() for _ in range(BATCH_SIZE)]

    for j in range(1, len(images)):
        drawRandomTriangle(images[j], colors)

    fit = [getFitness(ref, img) for img in images]

    min_ = 100
    winner = -1
    for j in range(len(fit)):
        if fit[j] < min_:
            min_ = fit[j]
            winner = j

    base = images[winner]
    print(f"Epoch: {i}, Fitness: {fit[winner]}")
    if min_ != last:
        images[winner].save(f"Steps\\Epoch{i}.jpg")
    last = min_

print("--- %s seconds ---" % (time.time() - start_time))
fig = plt.figure(figsize=(2, 2))
fig.add_subplot(2, 2, 1)
plt.imshow(ref)
fig.add_subplot(2, 2, 2)
plt.imshow(images[winner])
fig.add_subplot(2, 2, 3)
plt.imshow(images[0])
fig.add_subplot(2, 2, 4)
plt.imshow(images[1])
plt.show()

# plt.imshow(images[winner])
