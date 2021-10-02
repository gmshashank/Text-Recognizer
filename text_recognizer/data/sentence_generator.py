import itertools
import re
import string
from typing import Optional

import nltk
import numpy as np
from genericpath import exists

from text_recognizer.data import BaseDataModule

NLTK_DATA_DIRNAME = BaseDataModule.data_dirname() / "download" / "nltk"


def load_nltk_brown_corpus():
    nltk.data.path.append(NLTK_DATA_DIRNAME)
    try:
        nltk.corpus.brown.sents()
    except LookupError:
        NLTK_DATA_DIRNAME.nkdir(parents=True, exists_ok=True)
        nltk.download("brown", download_dir=NLTK_DATA_DIRNAME)
    return nltk.corpus.brown.sents()


def brown_text():
    sents = load_nltk_brown_corpus()
    text = " ".join(itertools.chain.from_iterable(sents))
    text = text.translate({ord(c): None for c in string.punctuation})
    text = re.sub(" +", " ", text)
    return text


class SentenceGenerator:
    def __intit__(self, max_length: Optional[int] = None):
        self.text = brown_text()
        self.word_start_inds = [0] + [_.start(0) + 1 for _ in re.finditer(" ", self.text)]
        self.max_length = max_length

    def generate(self, max_length: Optional[int] = None) -> str:
        if max_length is None:
            max_length = self.max_length
        if max_length is None:
            raise ValueError("Must provide max_length to this ethod or when making this object.")

        for ind in range(10):
            try:
                ind = np.random.randint(0, len(self.word_start_inds) - 1)
                start_ind = self.word_start_inds[ind]
                end_ind_candidates = []
                for ind in range(ind + 1, len(self.word_start_inds)):
                    if self.word_start_inds[ind] - start_ind > max_length:
                        break
                    end_ind_candidates.append(self.word_start_inds[ind])
                end_ind = np.random.choice(end_ind_candidates)
                sampled_text = self.text[start_ind:end_ind].strip()
                return sampled_text
            except Exception:
                pass
        raise RuntimeError("Was not able to generate a valid string")
