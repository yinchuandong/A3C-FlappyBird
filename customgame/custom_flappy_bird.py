import os
import sys
import numpy as np
import random
import pygame
import pygame.surfarray as surfarray
# from pygame.locals import *
from itertools import cycle


class CustomFlappyBird(object):
    def __init__(self, fps=60, screen_width=288, screen_height=512, display_screen=True, frame_skip=1):
        pygame.init()
        self._fps = fps
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._display_screen = display_screen
        self._frame_skip = frame_skip

        self._fps_clock = pygame.time.Clock()
        self._screen = pygame.display.set_mode((self._screen_width, self._screen_height))
        pygame.display.set_caption('Flappy Bird')

        self._images, self._sounds, self._hit_masks = self._load_resources()
        self._pip_gap_size = 100  # gap between upper and lower part of pipe
        self._basey = self._screen_height * 0.79

        self._player_width = self._images['player'][0].get_width()
        self._player_height = self._images['player'][0].get_height()
        self._pipe_width = self._images['pipe'][0].get_width()
        self._pip_height = self._images['pipe'][0].get_height()
        self._bg_width = self._images['background'].get_width()

        self.reset()
        return

    def _new_game(self):
        self._player_index_gen = cycle([0, 1, 2, 1])

        self._score = self._player_index = self._loop_iter = 0
        self._player_x = int(self._screen_width * 0.2)
        self._player_y = int((self._screen_height - self._player_height) / 2)
        self._base_x = 0
        self._base_shift = self._images[
            'base'].get_width() - self._bg_width

        newPipe1 = self._get_random_pipe()
        newPipe2 = self._get_random_pipe()
        self._upper_pipes = [
            {'x': self._screen_width, 'y': newPipe1[0]['y']},
            {'x': self._screen_width + (self._screen_width / 2), 'y': newPipe2[0]['y']},
        ]
        self._lower_pipes = [
            {'x': self._screen_width, 'y': newPipe1[1]['y']},
            {'x': self._screen_width + (self._screen_width / 2), 'y': newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on
        # flap
        self._pipe_vel_x = -4
        self._player_vel_y = 0    # player's velocity along Y, default same as _player_flapped
        self._player_max_vel_x = 10   # max vel along Y, max descend speed
        self._player_min_vel_y = -8   # min vel along Y, max ascend speed
        self._player_acc_y = 1   # players downward accleration
        self._player_flap_acc = -9   # players speed on flapping
        self._player_flapped = False  # True when player flaps
        return

    def _frame_step(self, input_actions):
        pygame.event.pump()

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:
            if self._player_y > -2 * self._player_height:
                self._player_vel_y = self._player_flap_acc
                self._player_flapped = True
                # self._sounds['wing'].play()

        # check for score
        playerMidPos = self._player_x + self._player_width / 2
        for pipe in self._upper_pipes:
            pipeMidPos = pipe['x'] + self._pipe_width / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self._score += 1
                # self._sounds['point'].play()
                reward = 1.0

        # _player_index basex change
        if (self._loop_iter + 1) % 3 == 0:
            self._player_index = next(self._player_index_gen)
        self._loop_iter = (self._loop_iter + 1) % 30
        self._base_x = -((-self._base_x + 100) % self._base_shift)

        # player's movement
        if self._player_vel_y < self._player_max_vel_x and not self._player_flapped:
            self._player_vel_y += self._player_acc_y
        if self._player_flapped:
            self._player_flapped = False
        self._player_y += min(self._player_vel_y, self._basey -
                              self._player_y - self._player_height)
        if self._player_y < 0:
            self._player_y = 0

        # move pipes to left
        for uPipe, lPipe in zip(self._upper_pipes, self._lower_pipes):
            uPipe['x'] += self._pipe_vel_x
            lPipe['x'] += self._pipe_vel_x

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self._upper_pipes[0]['x'] < 5:
            newPipe = self._get_random_pipe()
            self._upper_pipes.append(newPipe[0])
            self._lower_pipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self._upper_pipes[0]['x'] < -self._pipe_width:
            self._upper_pipes.pop(0)
            self._lower_pipes.pop(0)

        # check if crash here
        isCrash = self._check_crash({'x': self._player_x, 'y': self._player_y,
                                     'index': self._player_index},
                                    self._upper_pipes, self._lower_pipes)
        if isCrash:
            # self._sounds['hit'].play()
            # self._sounds['die'].play()
            terminal = True
            reward = -1.0
            # self.reset()

        # draw sprites
        self._screen.blit(self._images['background'], (0, 0))

        for uPipe, lPipe in zip(self._upper_pipes, self._lower_pipes):
            self._screen.blit(self._images['pipe'][0], (uPipe['x'], uPipe['y']))
            self._screen.blit(self._images['pipe'][1], (lPipe['x'], lPipe['y']))

        self._screen.blit(self._images['base'], (self._base_x, self._basey))
        # print score so player overlaps the score
        self._screen.blit(self._images['player'][self._player_index],
                          (self._player_x, self._player_y))

        img = self._capture_screen()

        if self._display_screen:
            pygame.display.update()
        self._fps_clock.tick(self._fps)
        # print self._upper_pipes[0]['y'] + self._pip_height - int(self._basey * 0.2)

        if terminal:
            self.reset()
        return img, reward, terminal

    def _capture_screen(self):
        img = surfarray.array3d(pygame.display.get_surface())
        return img

    def reset(self):
        self._new_game()
        o_t = self._capture_screen()
        return o_t

    def step(self, action_id):
        action = np.zeros([2])
        action[action_id] = 1

        total_reward = 0.0
        for _ in range(self._frame_skip):
            o_t1, reward, terminal = self._frame_step(action)
            total_reward += reward
            if terminal:
                break
        return o_t1, total_reward, terminal

    @property
    def action_size(self):
        return 2

    @property
    def action_set(self):
        return [0, 1]

    def _get_random_pipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
        index = random.randint(0, len(gapYs) - 1)
        gapY = gapYs[index]

        gapY += int(self._basey * 0.2)
        pipeX = self._screen_width + 10

        return [
            {'x': pipeX, 'y': gapY - self._pip_height},  # upper pipe
            {'x': pipeX, 'y': gapY + self._pip_gap_size},  # lower pipe
        ]

    def _check_crash(self, player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = self._images['player'][0].get_width()
        player['h'] = self._images['player'][0].get_height()

        # if player crashes into ground
        if player['y'] + player['h'] >= self._basey - 1:
            return True
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                                     player['w'], player['h'])

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(
                    uPipe['x'], uPipe['y'], self._pipe_width, self._pip_height)
                lPipeRect = pygame.Rect(
                    lPipe['x'], lPipe['y'], self._pipe_width, self._pip_height)

                # player and upper/lower pipe self.hit_masks_
                pHitMask = self._hit_masks['player'][pi]
                uHitmask = self._hit_masks['pipe'][0]
                lHitmask = self._hit_masks['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self._pixel_collision(
                    playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self._pixel_collision(
                    playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return True

        return False

    def _pixel_collision(self, rect1, rect2, hitmask1, hitmask2):
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

    def _load_resources(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        # path of player with different states
        player_path = (
            os.path.join(dir_path, 'assets/sprites/redbird-upflap.png'),
            os.path.join(dir_path, 'assets/sprites/redbird-midflap.png'),
            os.path.join(dir_path, 'assets/sprites/redbird-downflap.png')
        )

        # path of background
        background_path = os.path.join(dir_path, 'assets/sprites/background-black.png')
        # background_path = os.path.join(dir_path, 'assets/sprites/background-day.png')
        # background_path = os.path.join(dir_path, 'assets/sprites/background-night.png')

        # path of pipe
        PIPE_PATH = os.path.join(dir_path, 'assets/sprites/pipe-green.png')

        images, sounds, hit_masks = {}, {}, {}

        # numbers sprites for score display
        images['numbers'] = (
            pygame.image.load(os.path.join(dir_path, 'assets/sprites/0.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path, 'assets/sprites/1.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path, 'assets/sprites/2.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path, 'assets/sprites/3.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path, 'assets/sprites/4.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path, 'assets/sprites/5.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path, 'assets/sprites/6.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path, 'assets/sprites/7.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path, 'assets/sprites/8.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path, 'assets/sprites/9.png')).convert_alpha()
        )

        # base (ground) sprite
        images['base'] = pygame.image.load(os.path.join(dir_path, 'assets/sprites/base.png')).convert_alpha()

        # sounds
        if 'win' in sys.platform:
            soundExt = '.wav'
        else:
            soundExt = '.ogg'

        # sounds['die'] = pygame.mixer.Sound('assets/audio/die' + soundExt)
        # sounds['hit'] = pygame.mixer.Sound('assets/audio/hit' + soundExt)
        # sounds['point'] = pygame.mixer.Sound('assets/audio/point' + soundExt)
        # sounds['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
        # sounds['wing'] = pygame.mixer.Sound('assets/audio/wing' + soundExt)

        # select random background sprites
        images['background'] = pygame.image.load(background_path).convert()

        # select random player sprites
        images['player'] = (
            pygame.image.load(player_path[0]).convert_alpha(),
            pygame.image.load(player_path[1]).convert_alpha(),
            pygame.image.load(player_path[2]).convert_alpha(),
        )

        # select random pipe sprites
        images['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPE_PATH).convert_alpha(), 180),
            pygame.image.load(PIPE_PATH).convert_alpha(),
        )

        # hismask for pipes
        hit_masks['pipe'] = (
            self._get_hit_mask(images['pipe'][0]),
            self._get_hit_mask(images['pipe'][1]),
        )

        # hitmask for player
        hit_masks['player'] = (
            self._get_hit_mask(images['player'][0]),
            self._get_hit_mask(images['player'][1]),
            self._get_hit_mask(images['player'][2]),
        )

        return images, sounds, hit_masks

    def _get_hit_mask(self, image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x, y))[3]))
        return mask
