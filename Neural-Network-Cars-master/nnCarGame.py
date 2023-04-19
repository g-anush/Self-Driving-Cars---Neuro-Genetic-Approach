import pygame
import random
import math
import numpy as np
from datetime import datetime
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from utils import *


# ==================
# PYGAME CONSTANTS AND RESOURCES
# ==================

pygame.init()  # Initialize pygame
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
black = (0, 0, 0)
gray = pygame.Color("gray12")
Color_line = (255, 0, 0)

FPS = 30
lines = True  # If true then lines of player are shown
player = True  # If true then player is shown
display_info = True  # If true then display info is shown
frames = 0
maxspeed = 10

white_small_car = pygame.image.load("Images/Sprites/white_small.png")
white_big_car = pygame.image.load("Images/Sprites/white_big.png")
green_small_car = pygame.image.load("Images/Sprites/green_small.png")
green_big_car = pygame.image.load("Images/Sprites/green_big.png")

bg = pygame.image.load("train4front.png")
bg4 = pygame.image.load("train4back.png")

# These is just the text being displayed on pygame window
infoX = 1365
infoY = 600
font = pygame.font.Font("freesansbold.ttf", 18)
# text1 = font.render('0..9 - Change Mutation', True, white)
text1 = font.render("H - Load and Mate Best", True, white)
text2 = font.render("LMB - Select/Unselect", True, white)
text3 = font.render("RMB - Delete", True, white)
text4 = font.render("L - Show/Hide Lines", True, white)
text5 = font.render("R - Reset", True, white)
text6 = font.render("B - Breed Selected", True, white)
text7 = font.render("C - Clean", True, white)
text8 = font.render("N - Reverse Track", True, white)
text9 = font.render("A - Toggle Player", True, white)
text10 = font.render("D - Toggle Info", True, white)
text11 = font.render("M - Breed and Reverse Track", True, white)
text12 = font.render("T - Tough Track", True, white)
text13 = font.render("SPACE - Pause/Play", True, white)
text1Rect = text1.get_rect().move(infoX, infoY)
text2Rect = text2.get_rect().move(infoX, infoY + text1Rect.height)
text3Rect = text3.get_rect().move(infoX, infoY + 2 * text1Rect.height)
text4Rect = text4.get_rect().move(infoX, infoY + 3 * text1Rect.height)
text5Rect = text5.get_rect().move(infoX, infoY + 4 * text1Rect.height)
text6Rect = text6.get_rect().move(infoX, infoY + 5 * text1Rect.height)
text7Rect = text7.get_rect().move(infoX, infoY + 6 * text1Rect.height)
text8Rect = text8.get_rect().move(infoX, infoY + 7 * text1Rect.height)
text9Rect = text9.get_rect().move(infoX, infoY + 8 * text1Rect.height)
text10Rect = text10.get_rect().move(infoX, infoY + 9 * text1Rect.height)
text11Rect = text11.get_rect().move(infoX, infoY + 10 * text1Rect.height)
text12Rect = text12.get_rect().move(infoX, infoY + 11 * text1Rect.height)
text13Rect = text12.get_rect().move(infoX, infoY + 12 * text1Rect.height)


def displayTexts():
    infotextX = 20
    infotextY = 600
    infotext1 = font.render("Gen " + str(generation), True, white)
    infotext2 = font.render("Cars: " + str(num_of_nnCars), True, white)
    infotext3 = font.render("Alive: " + str(alive), True, white)
    infotext4 = font.render("Selected: " + str(selected), True, white)
    if lines == True:
        infotext5 = font.render("Lines ON", True, white)
    else:
        infotext5 = font.render("Lines OFF", True, white)
    if player == True:
        infotext6 = font.render("Player ON", True, white)
    else:
        infotext6 = font.render("Player OFF", True, white)
    # infotext7 = font.render('Mutation: '+ str(2*mutationRate), True, white)
    # infotext8 = font.render('Frames: ' + str(frames), True, white)
    infotext9 = font.render("FPS: 30", True, white)
    infotext1Rect = infotext1.get_rect().move(infotextX, infotextY)
    infotext2Rect = infotext2.get_rect().move(
        infotextX, infotextY + infotext1Rect.height
    )
    infotext3Rect = infotext3.get_rect().move(
        infotextX, infotextY + 2 * infotext1Rect.height
    )
    infotext4Rect = infotext4.get_rect().move(
        infotextX, infotextY + 3 * infotext1Rect.height
    )
    infotext5Rect = infotext5.get_rect().move(
        infotextX, infotextY + 4 * infotext1Rect.height
    )
    infotext6Rect = infotext6.get_rect().move(
        infotextX, infotextY + 5 * infotext1Rect.height
    )
    # infotext7Rect = infotext7.get_rect().move(infotextX,infotextY+6*infotext1Rect.height)
    # infotext8Rect = infotext8.get_rect().move(infotextX,infotextY+7*infotext1Rect.height)
    infotext9Rect = infotext9.get_rect().move(
        infotextX, infotextY + 6 * infotext1Rect.height
    )

    gameDisplay.blit(text1, text1Rect)
    gameDisplay.blit(text2, text2Rect)
    gameDisplay.blit(text3, text3Rect)
    gameDisplay.blit(text4, text4Rect)
    gameDisplay.blit(text5, text5Rect)
    gameDisplay.blit(text6, text6Rect)
    gameDisplay.blit(text7, text7Rect)
    gameDisplay.blit(text8, text8Rect)
    gameDisplay.blit(text9, text9Rect)
    gameDisplay.blit(text10, text10Rect)
    gameDisplay.blit(text11, text11Rect)
    gameDisplay.blit(text12, text12Rect)
    gameDisplay.blit(text13, text13Rect)

    gameDisplay.blit(infotext1, infotext1Rect)
    gameDisplay.blit(infotext2, infotext2Rect)
    gameDisplay.blit(infotext3, infotext3Rect)
    gameDisplay.blit(infotext4, infotext4Rect)
    gameDisplay.blit(infotext5, infotext5Rect)
    gameDisplay.blit(infotext6, infotext6Rect)
    # gameDisplay.blit(infotext7, infotext7Rect)
    # gameDisplay.blit(infotext8, infotext8Rect)
    gameDisplay.blit(infotext9, infotext9Rect)
    return


# ==================
# GENETIC ALGORITHM AND IMPLEMENTATION
# ==================


class Car:
    def __init__(self, sizes):
        self.score = 0
        self.num_layers = len(sizes)  # Number of nn layers
        self.sizes = sizes  # List with number of neurons per layer
        self.biases = [
            np.random.randn(y, 1) for y in sizes[1:]
        ]  # Bias to be added to layer[i] is biases[i-1]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])
        ]  # Weight for layer[i] is weights[i]

        # c1, c2, c3, c4, c5 are five 2D points where the car could collided, updated in every frame
        self.c1 = 0, 0
        self.c2 = 0, 0
        self.c3 = 0, 0
        self.c4 = 0, 0
        self.c5 = 0, 0
        # d1, d2, d3, d4, d5 are distances from the car to those points, updated every frame too and used as the input for the NN
        self.d1 = 0
        self.d2 = 0
        self.d3 = 0
        self.d4 = 0
        self.d5 = 0
        self.dead = False

        # The input and output of the NN must be in a numpy array format
        self.inp = np.array([[self.d1], [self.d2], [self.d3], [self.d4], [self.d5]])
        self.outp = np.array([[0], [0], [0], [0]])

        # Boolean used for toggling distance lines
        self.showlines = False

        # Initial location of the car
        self.x = 120
        self.y = 480
        self.center = self.x, self.y

        # Height and width of the car
        self.height = 35  # 45
        self.width = 17  # 25

        # These are the four corners of the car, using polygon instead of rectangle object, when rotating or moving the car, we rotate or move these
        self.d = self.x - (self.width / 2), self.y - (self.height / 2)
        self.c = self.x + self.width - (self.width / 2), self.y - (self.height / 2)
        self.b = self.x + self.width - (self.width / 2), self.y + self.height - (
            self.height / 2
        )
        self.a = self.x - (self.width / 2), self.y + self.height - (self.height / 2)

        # Velocity, acceleration and direction of the car
        self.velocity = 0
        self.acceleration = 0
        self.angle = 180

        # Boolean which goes true when car collides
        self.collided = False
        # Car color and image
        self.color = white
        self.car_image = white_small_car

    def set_accel(self, accel):
        self.acceleration = accel

    def rotate(self, rot):
        self.angle += rot
        if self.angle > 360:
            self.angle = 0
        if self.angle < 0:
            self.angle = 360 + self.angle

    def update(self):
        self.score += self.velocity
        if self.acceleration != 0:
            self.velocity += self.acceleration
            if self.velocity > maxspeed:
                self.velocity = maxspeed
            elif self.velocity < 0:
                self.velocity = 0
        else:
            self.velocity *= 0.92

        self.x, self.y = move((self.x, self.y), self.angle, self.velocity)
        self.center = self.x, self.y

        self.d = self.x - (self.width / 2), self.y - (self.height / 2)
        self.c = self.x + self.width - (self.width / 2), self.y - (self.height / 2)
        self.b = self.x + self.width - (self.width / 2), self.y + self.height - (
            self.height / 2
        )
        self.a = self.x - (self.width / 2), self.y + self.height - (self.height / 2)

        self.a = rotation((self.x, self.y), self.a, math.radians(self.angle))
        self.b = rotation((self.x, self.y), self.b, math.radians(self.angle))
        self.c = rotation((self.x, self.y), self.c, math.radians(self.angle))
        self.d = rotation((self.x, self.y), self.d, math.radians(self.angle))

        self.c1 = move((self.x, self.y), self.angle, 10)
        while bg4.get_at((int(self.c1[0]), int(self.c1[1]))).a != 0:
            self.c1 = move((self.c1[0], self.c1[1]), self.angle, 10)
        while bg4.get_at((int(self.c1[0]), int(self.c1[1]))).a == 0:
            self.c1 = move((self.c1[0], self.c1[1]), self.angle, -1)

        self.c2 = move((self.x, self.y), self.angle + 45, 10)
        while bg4.get_at((int(self.c2[0]), int(self.c2[1]))).a != 0:
            self.c2 = move((self.c2[0], self.c2[1]), self.angle + 45, 10)
        while bg4.get_at((int(self.c2[0]), int(self.c2[1]))).a == 0:
            self.c2 = move((self.c2[0], self.c2[1]), self.angle + 45, -1)

        self.c3 = move((self.x, self.y), self.angle - 45, 10)
        while bg4.get_at((int(self.c3[0]), int(self.c3[1]))).a != 0:
            self.c3 = move((self.c3[0], self.c3[1]), self.angle - 45, 10)
        while bg4.get_at((int(self.c3[0]), int(self.c3[1]))).a == 0:
            self.c3 = move((self.c3[0], self.c3[1]), self.angle - 45, -1)

        self.c4 = move((self.x, self.y), self.angle + 90, 10)
        while bg4.get_at((int(self.c4[0]), int(self.c4[1]))).a != 0:
            self.c4 = move((self.c4[0], self.c4[1]), self.angle + 90, 10)
        while bg4.get_at((int(self.c4[0]), int(self.c4[1]))).a == 0:
            self.c4 = move((self.c4[0], self.c4[1]), self.angle + 90, -1)

        self.c5 = move((self.x, self.y), self.angle - 90, 10)
        while bg4.get_at((int(self.c5[0]), int(self.c5[1]))).a != 0:
            self.c5 = move((self.c5[0], self.c5[1]), self.angle - 90, 10)
        while bg4.get_at((int(self.c5[0]), int(self.c5[1]))).a == 0:
            self.c5 = move((self.c5[0], self.c5[1]), self.angle - 90, -1)

        self.d1 = int(
            calculateDistance(self.center[0], self.center[1], self.c1[0], self.c1[1])
        )
        self.d2 = int(
            calculateDistance(self.center[0], self.center[1], self.c2[0], self.c2[1])
        )
        self.d3 = int(
            calculateDistance(self.center[0], self.center[1], self.c3[0], self.c3[1])
        )
        self.d4 = int(
            calculateDistance(self.center[0], self.center[1], self.c4[0], self.c4[1])
        )
        self.d5 = int(
            calculateDistance(self.center[0], self.center[1], self.c5[0], self.c5[1])
        )

    def draw(self, display):
        rotated_image = pygame.transform.rotate(self.car_image, -self.angle - 180)
        rect_rotated_image = rotated_image.get_rect()
        rect_rotated_image.center = self.x, self.y
        gameDisplay.blit(rotated_image, rect_rotated_image)

        center = self.x, self.y
        if self.showlines:
            pygame.draw.line(gameDisplay, Color_line, (self.x, self.y), self.c1, 2)
            pygame.draw.line(gameDisplay, Color_line, (self.x, self.y), self.c2, 2)
            pygame.draw.line(gameDisplay, Color_line, (self.x, self.y), self.c3, 2)
            pygame.draw.line(gameDisplay, Color_line, (self.x, self.y), self.c4, 2)
            pygame.draw.line(gameDisplay, Color_line, (self.x, self.y), self.c5, 2)

    def toggleShowLines(self):
        self.showlines = not self.showlines

    def feedforward(self):
        # Return the output of the network
        self.inp = np.array(
            [[self.d1], [self.d2], [self.d3], [self.d4], [self.d5], [self.velocity]]
        )
        for b, w in zip(self.biases, self.weights):
            self.inp = sigmoid(np.dot(w, self.inp) + b)

        self.outp = self.inp
        return self.outp

    def collision(self):
        if (
            (bg4.get_at((int(self.a[0]), int(self.a[1]))).a == 0)
            or (bg4.get_at((int(self.b[0]), int(self.b[1]))).a == 0)
            or (bg4.get_at((int(self.c[0]), int(self.c[1]))).a == 0)
            or (bg4.get_at((int(self.d[0]), int(self.d[1]))).a == 0)
        ):
            return True
        else:
            return False

    def resetPosition(self):
        if number_track == 0:
            self.x = 120
            self.y = 480
            self.angle = 180
            if not test_track:
                self.x = 610
                self.y = 710

        if number_track == 1:
            self.x = 1470
            self.y = 455
            self.angle = 180

            if not test_track:
                self.x = 1210
                self.y = 470
                self.angle = 90

        return

    def takeAction(self):
        if self.outp.item(0) > 0.5:  # Accelerate
            self.set_accel(0.2)
        else:
            self.set_accel(0)
        if self.outp.item(1) > 0.5:  # Brake
            self.set_accel(-0.2)
        if self.outp.item(2) > 0.5:  # Turn right
            self.rotate(-5)
        if self.outp.item(3) > 0.5:  # Turn left
            self.rotate(5)
        return


generation = 1
mutationRate = 90
mutationProb = 0.5
mutationProbPerGene = 0.2
selectedCars = []
selected = 0
number_track = 0
test_track = False

nnCars = []  # List of neural network cars
num_of_nnCars = 200  # Number of neural network cars
alive = num_of_nnCars  # Number of not collided (alive) cars
collidedCars = []  # List containing collided cars

inputLayer = 6
hiddenLayer = 6
outputLayer = 4


def mutateWeightGene(parent1, child1):
    sizenn = len(child1.sizes)

    # Copy parent weights into child weights
    for i in range(sizenn - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]

    # Copy parent biases into child biases
    for i in range(sizenn - 1):
        for j in range(child1.sizes[i + 1]):
            child1.biases[i][j] = parent1.biases[i][j]

    genomeWeights = (
        []
    )  # This will be a list containing all weights, easier to modify this way

    for i in range(sizenn - 1):  # i=0,1
        for j in range(child1.sizes[i] * child1.sizes[i + 1]):
            genomeWeights.append(child1.weights[i].item(j))

    for w in range(len(genomeWeights)):
        if random.random() < mutationProbPerGene:
            genomeWeights[w] = genomeWeights[w] * random.uniform(0.7, 1.3)
        if random.random() < 0.01:
            genomeWeights[w] = genomeWeights[w] * -1

    count = 0
    for i in range(sizenn - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = genomeWeights[count]
                count += 1
    return


def mutateBiasesGene(parent1, child1):
    sizenn = len(child1.sizes)

    for i in range(sizenn - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]

    for i in range(sizenn - 1):
        for j in range(child1.sizes[i + 1]):
            child1.biases[i][j] = parent1.biases[i][j]

    genomeBiases = []

    for i in range(sizenn - 1):
        for j in range(child1.sizes[i + 1]):
            genomeBiases.append(child1.biases[i].item(j))

    for b in range(len(genomeBiases)):
        if random.random() < mutationProbPerGene:
            genomeBiases[b] = genomeBiases[b] * random.uniform(0.7, 1.3)

    count = 0
    for i in range(sizenn - 1):
        for j in range(child1.sizes[i + 1]):
            child1.biases[i][j] = genomeBiases[count]
            count += 1
    return


def uniformCrossOverWeights(
    parent1, parent2, child1, child2
):  # Given two parent car objects, it modifies the children car objects weights
    sizenn = len(child1.sizes)  # 3 si car1=Car([2, 4, 3])

    # Copy parent weights into child weights
    for i in range(sizenn - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]

    for i in range(sizenn - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child2.weights[i][j][k] = parent2.weights[i][j][k]

    # Copy parent biases into child biases
    for i in range(sizenn - 1):
        for j in range(child2.sizes[i + 1]):
            child1.biases[i][j] = parent1.biases[i][j]

    for i in range(sizenn - 1):
        for j in range(child2.sizes[i + 1]):
            child2.biases[i][j] = parent2.biases[i][j]

    genome1 = []  # This will be a list containing all weights of child1
    genome2 = []  # This will be a list containing all weights of child2

    for i in range(sizenn - 1):  # i=0,1
        for j in range(child1.sizes[i] * child1.sizes[i + 1]):
            genome1.append(child1.weights[i].item(j))

    for i in range(sizenn - 1):  # i=0,1
        for j in range(child2.sizes[i] * child2.sizes[i + 1]):
            genome2.append(child2.weights[i].item(j))

    # Crossover weights
    alter = True
    for i in range(len(genome1)):
        alter = random.random() < 0.5
        if alter == True:
            aux = genome1[i]
            genome1[i] = genome2[i]
            genome2[i] = aux

    # Go back from genome list to weights numpy array on child object
    count = 0
    for i in range(sizenn - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = genome1[count]
                count += 1

    count = 0
    for i in range(sizenn - 1):
        for j in range(child2.sizes[i + 1]):
            for k in range(child2.sizes[i]):
                child2.weights[i][j][k] = genome2[count]
                count += 1
    return


def uniformCrossOverBiases(
    parent1, parent2, child1, child2
):  # Given two parent car objects, it modifies the children car objects biases
    sizenn = len(parent1.sizes)

    for i in range(sizenn - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child1.weights[i][j][k] = parent1.weights[i][j][k]

    for i in range(sizenn - 1):
        for j in range(child1.sizes[i + 1]):
            for k in range(child1.sizes[i]):
                child2.weights[i][j][k] = parent2.weights[i][j][k]

    for i in range(sizenn - 1):
        for j in range(child2.sizes[i + 1]):
            child1.biases[i][j] = parent1.biases[i][j]

    for i in range(sizenn - 1):
        for j in range(child2.sizes[i + 1]):
            child2.biases[i][j] = parent2.biases[i][j]

    genome1 = []
    genome2 = []

    for i in range(sizenn - 1):
        for j in range(child1.sizes[i + 1]):
            genome1.append(child1.biases[i].item(j))

    for i in range(sizenn - 1):
        for j in range(child2.sizes[i + 1]):
            genome2.append(child2.biases[i].item(j))

    alter = True
    for i in range(len(genome1)):
        alter = random.random() < 0.5
        if alter == True:
            aux = genome1[i]
            genome1[i] = genome2[i]
            genome2[i] = aux

    count = 0
    for i in range(sizenn - 1):
        for j in range(child1.sizes[i + 1]):
            child1.biases[i][j] = genome1[count]
            count += 1

    count = 0
    for i in range(sizenn - 1):
        for j in range(child2.sizes[i + 1]):
            child2.biases[i][j] = genome2[count]
            count += 1
    return


def perform_mating():
    global alive
    global selected
    global generation
    global nnCars
    global selectedCars

    if len(selectedCars) != 2:
        return

    save_selected_cars()

    for nncar in nnCars:
        nncar.score = 0

    alive = num_of_nnCars
    generation += 1
    selected = 0
    nnCars.clear()

    for i in range(num_of_nnCars):
        nnCars.append(Car([inputLayer, hiddenLayer, outputLayer]))

    for i in range(0, num_of_nnCars - 2, 2):
        uniformCrossOverWeights(
            selectedCars[0], selectedCars[1], nnCars[i], nnCars[i + 1]
        )
        uniformCrossOverBiases(
            selectedCars[0], selectedCars[1], nnCars[i], nnCars[i + 1]
        )

    nnCars[num_of_nnCars - 2] = selectedCars[0]
    nnCars[num_of_nnCars - 1] = selectedCars[1]

    nnCars[num_of_nnCars - 2].car_image = green_small_car
    nnCars[num_of_nnCars - 1].car_image = green_small_car

    nnCars[num_of_nnCars - 2].resetPosition()
    nnCars[num_of_nnCars - 1].resetPosition()

    nnCars[num_of_nnCars - 2].collided = False
    nnCars[num_of_nnCars - 1].collided = False

    for i in range(num_of_nnCars - 2):
        # for j in range(mutationRate):
        if random.random() < mutationProb:
            mutateWeightGene(nnCars[i], auxcar)
            mutateWeightGene(auxcar, nnCars[i])
            mutateBiasesGene(nnCars[i], auxcar)
            mutateBiasesGene(auxcar, nnCars[i])

    for nncar in nnCars:
        nncar.velocity = 0
        nncar.acceleration = 0
        nncar.collided = False
        nncar.resetPosition()

    selectedCars.clear()


def redrawGameWindow():  # Called on very frame

    global alive
    global frames

    frames += 1

    gameD = gameDisplay.blit(bg, (0, 0))

    # NN cars
    # if not paused:
    for nncar in nnCars:
        if not paused:
            if not nncar.collided:
                nncar.update()  # Update: Every car center coord, corners, directions, collision points and collision distances

            if nncar.collision():  # Check which car collided
                nncar.collided = (
                    True  # If collided then change collided attribute to true
                )
                if nncar.dead == False:
                    alive -= 1
                    nncar.dead = True

            else:  # If not collided then feedforward the input and take an action
                nncar.feedforward()
                nncar.takeAction()
        nncar.draw(gameDisplay)

    # Same but for player
    if player:
        car.update()
        if car.collision():
            car.resetPosition()
            car.update()
        car.draw(gameDisplay)
    if display_info:
        displayTexts()
    pygame.display.update()  # updates the screen


def save_selected_cars():
    with open("cars.npy", "wb") as f:
        np.save(f, selectedCars[0].weights)
        np.save(f, selectedCars[0].biases)
        np.save(f, selectedCars[1].weights)
        np.save(f, selectedCars[1].biases)


def load_best_cars():
    car1 = Car([inputLayer, hiddenLayer, outputLayer])
    car2 = Car([inputLayer, hiddenLayer, outputLayer])
    with open("bestcars.npy", "rb") as f:
        car1.weights = np.load(f, allow_pickle=True)
        car2.biases = np.load(f, allow_pickle=True)
        car1.weights = np.load(f, allow_pickle=True)
        car2.biases = np.load(f, allow_pickle=True)
    selectedCars.clear()
    selectedCars.append(car1)
    selectedCars.append(car2)
    perform_mating()


# =============
# MAIN CODE
# =============


size = width, height = 1600, 900  # Size to use when creating pygame window
gameDisplay = pygame.display.set_mode(size)  # creates screen
clock = pygame.time.Clock()

car = Car([inputLayer, hiddenLayer, outputLayer])
auxcar = Car([inputLayer, hiddenLayer, outputLayer])
paused = True

for i in range(num_of_nnCars):
    nnCars.append(Car([inputLayer, hiddenLayer, outputLayer]))

for nncar in nnCars:
    nncar.resetPosition()

while True:

    for event in pygame.event.get():  # Check for events
        if event.type == pygame.QUIT:
            pygame.quit()  # quits
            quit()

        if event.type == pygame.KEYDOWN:  # If user uses the keyboard
            if event.key == ord("l"):  # toggle players sensor lines
                car.toggleShowLines()
                lines = not lines

            if event.key == ord("c"):  # clear dead cars
                for nncar in nnCars:
                    if nncar.collided == True:
                        nnCars.remove(nncar)
                        if nncar.dead == False:
                            alive -= 1

            if event.key == ord("a"):  # toggle player
                player = not player

            if event.key == ord("d"):  # toggle display info
                display_info = not display_info

            if event.key == ord("n"):  # N : next track without crossover
                number_track = (number_track + 1) % 2
                for nncar in nnCars:
                    nncar.velocity = 0
                    nncar.acceleration = 0
                    nncar.resetPosition()
                    nncar.collided = False

            if event.key == ord("t"):  # T : toggle Test track
                for nncar in nnCars:
                    nncar.velocity = 0
                    nncar.acceleration = 0
                    nncar.resetPosition()
                    nncar.collided = False

                if not test_track:
                    bg = pygame.image.load("track1front.png")
                    bg4 = pygame.image.load("track1back.png")
                else:
                    bg = pygame.image.load("train4front.png")
                    bg4 = pygame.image.load("train4back.png")

                test_track = not test_track
                for nncar in nnCars:
                    nncar.resetPosition()

            if event.key == ord("b"):  # perform crossover on selected
                perform_mating()

            if event.key == ord("m"):  # crossover and next track

                if len(selectedCars) == 2:
                    number_track = (number_track + 1) % 2
                    perform_mating()

            if event.key == ord("h"):  # load best cars
                load_best_cars()

            if event.key == ord("r"):  # reset simulation
                generation = 1
                alive = num_of_nnCars
                nnCars.clear()
                selectedCars.clear()
                for i in range(num_of_nnCars):
                    nnCars.append(Car([inputLayer, hiddenLayer, outputLayer]))
                for nncar in nnCars:
                    nncar.resetPosition()

            if event.key == ord(" "):  # pause/play
                paused = not paused

        if event.type == pygame.MOUSEBUTTONDOWN:
            # This returns a tuple:
            # (leftclick, middleclick, rightclick)
            # Each one is a boolean integer representing button up/down.
            mouses = pygame.mouse.get_pressed()
            if mouses[0]:
                pos = pygame.mouse.get_pos()
                point = Point(pos[0], pos[1])
                # Revisar la lista de autos y ver cual estaba ahi
                for nncar in nnCars:
                    polygon = Polygon([nncar.a, nncar.b, nncar.c, nncar.d])
                    if polygon.contains(point):
                        if nncar in selectedCars:
                            selectedCars.remove(nncar)
                            selected -= 1
                            if nncar.car_image == white_big_car:
                                nncar.car_image = white_small_car
                            if nncar.car_image == green_big_car:
                                nncar.car_image = green_small_car
                            if nncar.collided:
                                nncar.velocity = 0
                                nncar.acceleration = 0
                            nncar.update()
                        else:
                            if len(selectedCars) < 2:
                                selectedCars.append(nncar)
                                selected += 1
                                if nncar.car_image == white_small_car:
                                    nncar.car_image = white_big_car
                                if nncar.car_image == green_small_car:
                                    nncar.car_image = green_big_car
                                if nncar.collided:
                                    nncar.velocity = 0
                                    nncar.acceleration = 0
                                nncar.update()
                        break

            if mouses[2]:
                pos = pygame.mouse.get_pos()
                point = Point(pos[0], pos[1])
                for nncar in nnCars:
                    polygon = Polygon([nncar.a, nncar.b, nncar.c, nncar.d])
                    if polygon.contains(point):
                        if nncar not in selectedCars:
                            nnCars.remove(nncar)
                            alive -= 1
                        break

    # player controls
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        car.rotate(-5)
    if keys[pygame.K_RIGHT]:
        car.rotate(5)
    if keys[pygame.K_UP]:
        car.set_accel(0.2)
    else:
        car.set_accel(0)
    if keys[pygame.K_DOWN]:
        car.set_accel(-0.2)

    redrawGameWindow()

    clock.tick(FPS)
