import numpy as np
import sys
import random
import pygame
import flappy_bird_utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

from PIL import Image

FPS = 60
SCREENWIDTH = 288
SCREENHEIGHT = 512


class GameState:
    def __init__(self):
        pygame.init()
        self.FPSCLOCK = pygame.time.Clock()
        self.SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
        pygame.display.set_caption('Flappy Bird')

        self.IMAGES, SOUNDS, self.HITMASKS = flappy_bird_utils.load()
        self.PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
        self.BASEY = SCREENHEIGHT * 0.79

        self.PLAYER_WIDTH = self.IMAGES['player'][0].get_width()
        self.PLAYER_HEIGHT = self.IMAGES['player'][0].get_height()
        self.PIPE_WIDTH = self.IMAGES['pipe'][0].get_width()
        self.PIPE_HEIGHT = self.IMAGES['pipe'][0].get_height()
        self.BACKGROUND_WIDTH = self.IMAGES['background'].get_width()

        self.PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - self.PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = self.IMAGES[
            'base'].get_width() - self.BACKGROUND_WIDTH

        newPipe1 = self.getRandomPipe()
        newPipe2 = self.getRandomPipe()
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on
        # flap
        self.pipeVelX = -4
        self.playerVelY = 0    # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10   # max vel along Y, max descend speed
        self.playerMinVelY = -8   # min vel along Y, max ascend speed
        self.playerAccY = 1   # players downward accleration
        self.playerFlapAcc = -9   # players speed on flapping
        self.playerFlapped = False  # True when player flaps
        self.reset()
        return

    def frame_step(self, input_actions):
        pygame.event.pump()

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:
            if self.playery > -2 * self.PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                # SOUNDS['wing'].play()

        # check for score
        playerMidPos = self.playerx + self.PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + self.PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                # SOUNDS['point'].play()
                reward = 1

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self.PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, self.BASEY -
                            self.playery - self.PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -self.PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check if crash here
        isCrash = self.checkCrash({'x': self.playerx, 'y': self.playery,
                              'index': self.playerIndex},
                             self.upperPipes, self.lowerPipes)
        if isCrash:
            # SOUNDS['hit'].play()
            # SOUNDS['die'].play()
            terminal = True
            self.__init__()
            reward = -1

        # draw sprites
        self.SCREEN.blit(self.IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        self.SCREEN.blit(self.IMAGES['base'], (self.basex, self.BASEY))
        # print score so player overlaps the score
        self.SCREEN.blit(self.IMAGES['player'][self.playerIndex],
                         (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        self.FPSCLOCK.tick(FPS)
        # print self.upperPipes[0]['y'] + self.PIPE_HEIGHT - int(self.BASEY *
        # 0.2)
        return image_data, reward, terminal

    def reset(self):
        action = np.zeros([2])
        action[0] = 1
        x_t, reward, terminal = self.frame_step(action)
        x_t = Image.fromarray(x_t).convert('L')
        x_t = x_t.resize((84, 84), Image.ANTIALIAS)
        self.s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        self.reward = reward
        self.terminal = terminal
        return

    def process(self, actionId):
        action = np.zeros([2])
        action[actionId] = 1
        x_t1, reward, terminal = self.frame_step(action)
        x_t1 = Image.fromarray(x_t1).convert('L')
        x_t1 = x_t1.resize((84, 84), Image.ANTIALIAS)
        x_t1 = np.reshape(x_t1, (84, 84, 1))
        self.s_t1 = np.append(self.s_t[:, :, 1:], x_t1, axis=2)
        self.reward = reward
        self.terminal = terminal
        return

    def update(self):
        self.s_t = self.s_t1
        return

    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
        index = random.randint(0, len(gapYs) - 1)
        gapY = gapYs[index]

        gapY += int(self.BASEY * 0.2)
        pipeX = SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - self.PIPE_HEIGHT},  # upper pipe
            {'x': pipeX, 'y': gapY + self.PIPEGAPSIZE},  # lower pipe
        ]

    def checkCrash(self, player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = self.IMAGES['player'][0].get_width()
        player['h'] = self.IMAGES['player'][0].get_height()

        # if player crashes into ground
        if player['y'] + player['h'] >= self.BASEY - 1:
            return True
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                                     player['w'], player['h'])

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(
                    uPipe['x'], uPipe['y'], self.PIPE_WIDTH, self.PIPE_HEIGHT)
                lPipeRect = pygame.Rect(
                    lPipe['x'], lPipe['y'], self.PIPE_WIDTH, self.PIPE_HEIGHT)

                # player and upper/lower pipe self.hitmasks
                pHitMask = self.HITMASKS['player'][pi]
                uHitmask = self.HITMASKS['pipe'][0]
                lHitmask = self.HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixelCollision(
                    playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self.pixelCollision(
                    playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return True

        return False

    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                    return True
        return False
