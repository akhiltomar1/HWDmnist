import pygame, sys
from pygame.locals import *
import numpy as np
from numpy import testing
from keras.models import load_model
import cv2


white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

IMGSAVE = False
model = load_model("bestmodel.h5")
labels = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

# Initialize app
pygame.init()
DISPLAY_SURFACE = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Digits")

# FONT = pygame.font.Font("freesansbold.tff", 18)

iswriting = False

number_xcord = []
number_ycord = []

BOUNDARYINC = 5
img_cnt = 1

PREDICT = True
#pygame.font.init()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAY_SURFACE, white, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_minx, rect_maxx = max(number_xcord[0] - BOUNDARYINC, 0), min(640, number_xcord[-1] + BOUNDARYINC)
            rect_miny, rect_maxy = max(number_ycord[0] - BOUNDARYINC, 0), min(number_ycord[-1] + BOUNDARYINC, 640)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAY_SURFACE))[rect_minx:rect_maxx, rect_miny:rect_maxy].T.astype(np.float32)

            if IMGSAVE:
                cv2.imwrite("image.png")
                img_cnt += 1

            if PREDICT:
                img = cv2.resize(img_arr, (28, 28))
                img = np.pad(img, (10, 10), 'constant', constant_values = 0)
                img = cv2.resize(img, (28, 28))/255

                label = str(labels[np.argmax(model.predict(img.reshape(1, 28, 28, 1)))])

                font = pygame.font.SysFont("Grobold",20)
                txt_surface = font.render(label, True, red, white)
                txt_rectobj = pygame.get_rect()

                txt_rectobj.left, txt_rectobj.bottom = rect_minx, rect_maxy

                DISPLAY_SURFACE.blit(txt_surface, txt_rectobj)

            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAY_SURFACE.fill(black)

        pygame.display.update()
