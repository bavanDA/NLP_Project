import sentencepiece as spm
from src.tokenization.dataset import Dataset
from src.constants import TOKENIZATION_TEST_FILE, TOKENIZATION_TRAIN_FILE, BASE_TOKENIZATION_URL


class Tokenizer:
    def __init__(self, vocab_size, data_url, model_type):
        self.model_type = model_type
        self.dataset = Dataset(data_url, 5)
        self.vocab_size = vocab_size

    def train(self, test_pieces):
        self.dataset.save_data(test_pieces)
        spm.SentencePieceTrainer.Train(
            f"input={TOKENIZATION_TRAIN_FILE} --vocab_size={self.vocab_size} --model_prefix={BASE_TOKENIZATION_URL} --model_type={self.model_type}")

    def evaluate_uknown_precentage(self):
        test_data = self.dataset.get_test_data()
        sp = spm.SentencePieceProcessor()
        sp.load(f'{BASE_TOKENIZATION_URL}.model')
        total_tokens = 0
        uknown_tokens = 0
        for sentence in test_data:
            ids = sp.EncodeAsIds(sentence)
            total_tokens += len(ids)
            uknown_tokens += len([id for id in ids if id == 0])
        return uknown_tokens / total_tokens * 100
