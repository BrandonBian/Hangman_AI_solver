
import argparse
import collections
import pandas as pd
import numpy as np
from model import RNN_model
import torch

def arg_parser():
    parser = argparse.ArgumentParser(description="hangman game config")
    parser.add_argument("--train_set", type=str, default="words_test.txt",
                        help="path of the train dictionary")
    parser.add_argument("--lives", type=int, default=6,
                        help="upper limit of fail guesses")
    args = parser.parse_args()
    return args


def load_model(model_path):
    model = RNN_model(target_dim=26, hidden_units=16)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

# Read in train data
full_dictionary_location = "words.txt"

def build_dictionary(dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary

words = build_dictionary(full_dictionary_location)

print(len(words))
print(words[:10])

print("===================")

# Clean all weird words
def retain_alphabets(input_string):
    result = ""
    for char in input_string:
        if char.isalpha():
            result += char.lower()
    return result

words_cleaned = []

for i in range(len(words)):
    cleaned = retain_alphabets(words[i])
    if len(cleaned) != 0:
        words_cleaned.append(cleaned)

# Group words by length
from collections import defaultdict

word_by_length = defaultdict(list)

for word in words_cleaned:
    word_by_length[(len(word))].append(word)



def patterns(word):
    pattern = defaultdict(list)

    for idx, char in enumerate(word):
        if char == '_':
            continue
        else:
            pattern[char].append(idx)
    
    return pattern


def potential_matches(word, pattern):
    search_space = word_by_length[len(word)]
    matches = []

    for word in search_space:
        is_match = True

        for char, indices in pattern.items():
            for idx in indices:
                if word[idx] != char or any(word[i] == char for i in range(len(word)) if i not in indices):
                    is_match = False
                    break

            if not is_match:
                break

        if is_match:
            matches.append(word)

    return matches


def matches_top_freq(matches):
    all_chars = ''.join(matches)
    char_count = {}
    
    for char in all_chars:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    
    freq = [(char, count) for char, count in char_count.items()]
    return sorted(freq, key=lambda x: x[1], reverse=True)

########################################

class HangmanGame(object):
    def __init__(self, train_set_path, model_path="model.pth"):
        self.guessed_letters = []
        full_dictionary_location = train_set_path
        self.full_dictionary = self.build_dictionary(full_dictionary_location)
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        self.current_dictionary = []
        self.model = load_model(model_path)

    def encode_obscure_words(self, word):
        word_idx = [ord(i) - 97 if i != "_" else 26 for i in word]
        obscured_word = np.zeros((len(word), 27), dtype=np.float32)
        for i, j in enumerate(word_idx):
            obscured_word[i, j] = 1
        print("Obs:", obscured_word)

    def guess(self, word):  # word input example: "_ p p _ e "
        # first guess by letter frequency in each word group
        new_condition = patterns(''.join(word))

        search_space = potential_matches(''.join(word), new_condition)
        freq = matches_top_freq(search_space)

        print(freq)
        for i in range(len(freq)):
            guess = freq[i][0]
            if freq[i][0] not in self.guessed_letters:
                return freq[i][0]

        print("here")
            
        # if we run out of 2-gram, use LSTM model to predict
        # the benefit of LSTM model is to add more uncertainty to the prediction
        guessed_multi_hot = np.zeros(26, dtype=np.float32)
        for letter in self.guessed_letters:
            idx = ord(letter) - 97
            guessed_multi_hot[idx] = 1.0

        obscure_words = self.encode_obscure_words(word)
        obscure_words = np.asarray(obscure_words)
        guessed_multi_hot = np.asarray(guessed_multi_hot)
        obscure_words = torch.from_numpy(obscure_words)
        guessed_multi_hot = torch.from_numpy(guessed_multi_hot)
        out = self.model(obscure_words, guessed_multi_hot)
        guess = torch.argmax(out, dim=2).item()
        guess = chr(guess + 97)
        return guess

    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location, "r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary

    def get_current_word(self):
        """
        combine target word and guessed letters to generate obscured word
        """
        word_seen = [letter if letter in self.guessed_letters else "_" for letter in self.target_word]
        return word_seen

    def start_game(self, num_lives=6, verbose=True):

        self.target_word = input("please enter a word for the computer to guess:")
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
        tries_remains = num_lives

        word_seen = self.get_current_word()
        if verbose:
            print("Successfully start a new game! # of tries remaining: {0}. Word: {1}.".format(tries_remains, word_seen))

        while tries_remains > 0:
            # get guessed letter from user code
            guess_letter = self.guess(word_seen)

            # append guessed letter to guessed letters field in hangman object
            self.guessed_letters.append(guess_letter)
            if verbose:
                print("Guessing letter: {0}".format(guess_letter))

            word_seen = self.get_current_word()
            print("current word:{}".format(word_seen))

            if "_" not in word_seen:
                print("Successfully finished game!! The word is:{}, {} tries left".format(word_seen, tries_remains))
                return True

            if guess_letter not in self.target_word:
                tries_remains -= 1

        print("# of tries exceeded!")
        return False

if __name__ == "__main__":
    args = arg_parser()
    train_set = args.train_set
    game = HangmanGame(train_set)
    game.start_game(args.lives)