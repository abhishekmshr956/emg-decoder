import time

import numpy as np
from scipy.special import softmax
import pygame

from emg_decoder.src.visualization.key import Key

pygame.init()


class Keyboard:
    layout = [['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
              ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
              ['z', 'x', 'c', 'v', 'b', 'n', 'm'],
              [' '],
              [chr(0)]]

    def __init__(self):
        self.screen_size = (1000, 700)
        self.key_layout = [[Key(self.layout, char=char) for char in row] for row in self.layout]
        self.keyboard_size = (1000, 500)
        self.keyboard_start = (self.screen_size[0] - self.keyboard_size[0], self.screen_size[1] - self.keyboard_size[1])
        self.screen = pygame.display.set_mode(self.screen_size)
        self.font = pygame.font.Font('freesansbold.ttf', 24)
        self.layout_width = max([len(row) for row in self.layout])
        self.layout_height = len(self.layout)
        self.key_width = self.keyboard_size[0] / self.layout_width
        self.key_height = self.keyboard_size[1] / self.layout_height
        self.row_offsets = [row_idx * self.key_width * 0.4 for row_idx in range(3)]
        self.row_offsets.append(self.keyboard_size[0] / 2)
        self.row_offsets.append(self.keyboard_size[0] / 2)
        self.cold = (68, 1, 84)
        self.hot = (253, 231, 37)

        # Key border colors
        self.key_color = (255, 255, 255)
        self.default_key_border_color = (255, 255, 255)
        self.incorrect_key_border_color = (255, 0, 0)
        self.selected_key_border_color = (255, 255, 0)
        self.correct_key_border_color = (0, 255, 0)
        self.true_key_border_color = (0, 191, 255)
        self.border_width = 5

        self.probs = np.zeros(len(self.key_layout[0][0].flat_layout))
        self.current_top_k = None
        self.current_key_idx = None

    def run(self, probs, true_char_idx=None, k=3):
        """
        :param k: Top k keys to highlight
        :param probs: Softmaxed probabilities or raw logits of the next character
        :param true_char_idx: Index of the true character, if available
        :return:
        """
        assert(self.probs.size == probs.size)
        self.probs = softmax(probs)
        self.current_top_k = np.argpartition(probs, -k)[-k:]
        self.current_key_idx = np.argmax(probs)
        self.__draw_keyboard(true_char_idx=true_char_idx)
        self.__update_screen()

    def __draw_keyboard(self, true_char_idx=None):
        self.screen.fill((0, 0, 0))
        for row_idx, row in enumerate(self.layout):
            for col_idx, key in enumerate(row):
                key = self.key_layout[row_idx][col_idx]

                if row_idx == 3 and col_idx == 0:
                    key_rect = pygame.Rect(
                        self.keyboard_start[0] + 2 * self.key_width + 2 * self.key_width * 0.4,
                        self.keyboard_start[1] + row_idx * self.key_height,
                        self.key_width * 5,
                        self.key_height
                    )
                else:
                    key_rect = pygame.Rect(
                        self.keyboard_start[0] + col_idx * self.key_width + self.row_offsets[row_idx],
                        self.keyboard_start[1] + row_idx * self.key_height,
                        self.key_width,
                        self.key_height
                    )

                # Draw initial keys with heatmap color
                p_key = self.probs[self.key_layout[row_idx][col_idx].index]
                color = tuple([int(self.cold[i] + (self.hot[i] - self.cold[i]) * p_key) for i in range(3)])
                pygame.draw.rect(self.screen, color, key_rect)

                # Draw key borders
                pygame.draw.rect(self.screen, self.default_key_border_color, key_rect, width=1)
                # If the current key is the true key
                if key.index == true_char_idx:
                    pygame.draw.rect(self.screen, self.true_key_border_color, key_rect,
                                     width=self.border_width)
                # If the current key is one of the top k decoded keys
                if key.index in self.current_top_k:
                    pygame.draw.rect(self.screen, self.selected_key_border_color, key_rect,
                                     width=self.border_width)
                    if true_char_idx is not None:
                        if true_char_idx == key.index:  # If the true key is in the top k, we outline it green
                            pygame.draw.rect(self.screen, self.correct_key_border_color, key_rect,
                                             width=self.border_width)
                        elif true_char_idx not in self.current_top_k:
                            # If the true key is not in the top k, we outline them all red
                            pygame.draw.rect(self.screen, self.incorrect_key_border_color, key_rect,
                                             width=self.border_width)

                try:
                    key_text = self.font.render(key.char, True, self.key_color)
                except ValueError:
                    key_text = self.font.render("BLANK", True, self.key_color)

                self.screen.blit(key_text, key_text.get_rect(center=key_rect.center))

    @staticmethod
    def __update_screen():
        pygame.display.flip()


if __name__ == '__main__':
    kb = Keyboard()
    data_dir = "/Users/johnzhou/research/emg_decoder/data/processed/John-Zhou_2023-07-17_Open-Loop-Typing-Task"
    X_logits = np.load(f"{data_dir}/aug/x_logits.npy")
    y = np.load(f"{data_dir}/aug/y.npy")
    permuted_idxs = np.random.permutation(len(X_logits))
    for idx in permuted_idxs:
        # if np.argmax(X_logits[idx]) == y[idx]:
        #     continue
        # else:
        start = time.time()
        kb.run(X_logits[idx], true_char_idx=y[idx], k=1)
        print(f"Time: {(time.time() - start) * 1000} ms")
        time.sleep(0.5)
