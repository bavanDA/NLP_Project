
import json
from src.constants import TOKENIZATION_TEST_FILE, TOKENIZATION_TRAIN_FILE


class Dataset:
    def __init__(self, data_url, pieces_count) -> None:
        self.pieces_count = pieces_count
        with open(data_url) as f:
            self.__data = json.load(f)
        self.data_ready = False

    def save_data(self, test_pieces):
        step = len(self.__data) // self.pieces_count
        keys = list(self.__data.keys())
        with open(TOKENIZATION_TRAIN_FILE, 'w') as f:
            for i in range(len(keys)):
                current_piece = (i + 1) // step
                key = keys[i]
                if current_piece not in test_pieces:
                    f.write(f"{' '.join(self.__data[key])}\n")
        with open(TOKENIZATION_TEST_FILE, 'w') as f:
            for i in range(len(keys)):
                current_piece = (i + 1) // step
                key = keys[i]
                if current_piece in test_pieces:
                    f.write(f"{' '.join(self.__data[key])}\n")
        self.data_ready = True

    def get_test_data(self):
        if not self.data_ready:
            return None
        sentences = []
        with open(TOKENIZATION_TEST_FILE, 'r') as f:
            sentences = f.readlines()
        return sentences
