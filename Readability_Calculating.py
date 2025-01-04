import math
import os
import re
from pathlib import Path
import nltk
import numpy as np
from nltk.corpus import cmudict


class ReadingIndex(object):
    # https://readable.com/readability/
    # https://readabilityformulas.com/readability-scoring-system.php
    # https://www.textcompare.org/readability/

    def __init__(self,
                 text='',
                 verbose=True
                 ):
        self.text = text
        self.vocab = cmudict.dict()
        self.verbose = verbose
        self.reading_index = {}

    def get_syllables(self, word):
        try:
            syls = [syl for syl in self.vocab[word][0] if syl[-1].isdigit()]
        except:
            syls = []
        return syls

    def preprocess_text(self, text):
        text = re.sub("[，]", " , ", text)
        text = re.sub("[。]", " . ", text)
        text = re.sub("([a-zA-Z]+)\s+([,.])", "\\1\\2", text)  # comma & period should follow words closely
        text = re.sub("and,", "and", text)
        text = re.sub("or,", "and", text)
        text = re.sub("\\([^a-zA-Z0-9]+\\)", " ", text)   # remove non-english words
        text = re.sub("\s+", " ", text).strip()
        return text

    def compute_text_statistics(self, text):
        # clean text beforehand
        self.text = self.preprocess_text(text)
        # sentences
        self.sents = nltk.sent_tokenize(self.text)
        self.total_sents = len(self.sents)
        # words
        self.words = nltk.word_tokenize(self.text)
        self.words = [word.lower() for word in self.words if re.search('[0-9a-zA-Z]', word)]
        self.total_words = len(self.words)
        self.unique_words = set(self.words)
        self.total_unique_words = len(self.unique_words)
        # characters
        chars = [c for word in self.words for c in word]
        self.total_chars = len(chars)
        # syllables
        self.syllables = [(word, len(self.get_syllables(word))) for word in self.words]
        self.total_sylls = sum([ws[1] for ws in self.syllables])
        # word counts by syllable
        self.words_sylls = [[ws for ws in self.syllables if ws[1] == (i + 1)] for i in range(5)]
        self.words_sylls.append([ws for ws in self.syllables if ws[1] >= 6])
        self.total_words_by_syll = [len(x) for x in self.words_sylls]
        # # part of speech : nltk.help.upenn_tagset()
        # pos = nltk.pos_tag(words)
        # pos_n = [p for p in pos if re.search('^N', p[1])]
        # pos_v = [p for p in pos if re.search('^VB', p[1])]
        # pos_adj = [p for p in pos if re.search('^J', p[1])]
        # pos_adv = [p for p in pos if re.search('^RB', p[1])]
        # print(len(pos_n), len(pos_v), len(pos_adj), len(pos_adv))

        # self.total_sents = 28
        # self.total_words = 301
        # self.total_hdwds = 22
        # self.total_sylls = 401
        # self.total_chars = 1327
        self.words_per_sent = self.total_words / self.total_sents
        self.sends_per_word = self.total_sents / self.total_words
        self.sylls_per_word = self.total_sylls / self.total_words
        self.chars_per_word = self.total_chars / self.total_words
        self.total_ezwds = sum(self.total_words_by_syll[:2])
        self.total_hdwds = sum(self.total_words_by_syll[2:]) # polysyllables are words of 3 or more syllabels
        self.hdwds_ratio = self.total_hdwds / self.total_words

        if self.verbose:
            self.show_text_statistics()

    def show_text_statistics(self):
        print("sentence count:".ljust(30, ' '), self.total_sents)
        print("word count:".ljust(30, ' '), self.total_words)
        print("unique word count:".ljust(30, ' '), self.total_unique_words)
        print("syllable count:".ljust(30, ' '), self.total_sylls)
        print("character count:".ljust(30, ' '), self.total_chars)
        print("polysyllabic word count:".ljust(30, ' '), self.total_hdwds)
        print("average words per sentence:".ljust(30, ' '), round(self.words_per_sent, 2))
        print("average syllables per word:".ljust(30, ' '), round(self.sylls_per_word, 2))
        print("average characters per word:".ljust(30, ' '), round(self.chars_per_word, 2))

    def flesch_score2level(self, score):
        if score >= 90 and score < 100:
            level = 5
        elif score >= 80 and score < 90:
            level = 6
        elif score >= 70 and score < 80:
            level = 7
        elif score >= 60 and score < 70:
            level = 8
        elif score >= 50 and score < 60:
            level = 10
        elif score >= 30 and score < 50:
            level = 13
        elif score >= 10 and score < 30:
            level = 17
        elif score >= 0 and score < 10:
            level = 21
        else:
            level = 4
        return level

    def flesch_reading_ease(self):
        #  Flesch Reading Ease
        score = 206.835 - 1.015 * self.words_per_sent - 84.6 * self.sylls_per_word
        self.reading_index['flesch-ease-score'] = round(score, 2)
        self.reading_index['flesch-ease-level'] = self.flesch_score2level(score)
        if self.verbose:
            print(f"Flesch Reading Ease score:".ljust(35, ' '), round(score, 2))
            print(f"Flesch Reading Ease level:".ljust(35, ' '), self.reading_index['flesch-ease-level'])

    def automated_readability(self):
        # Automated Readability Index
        level = 4.71 * self.chars_per_word + 0.5 * self.words_per_sent - 21.43
        self.reading_index['automated-level'] = round(level, 2)
        if self.verbose:
            print(f"Automated Readability Index:".ljust(35, ' '), round(level, 2))

    def gunning_fog(self):
        # Gunning Fog Index
        level = 0.4 * (self.words_per_sent + 100 * self.hdwds_ratio)
        self.reading_index['gunning-level'] = round(level, 2)
        if self.verbose:
            print(f"Gunning Fog Index:".ljust(35, ' '), round(level, 2))

    def flesch_kincaid(self):
        #  Flesch-Kincaid Grade Level
        level = 0.39 * self.words_per_sent + 11.8 * self.sylls_per_word - 15.59
        self.reading_index['flesch-kincaid-level'] = round(level, 2)
        if self.verbose:
            print(f"Flesch-Kincaid Grade Level:".ljust(35, ' '), round(level, 2))

    def coleman_liau(self):
        # Coleman-Liau Index
        level = (0.0588 * 100 * self.chars_per_word) - (0.296 * 100 * self.sends_per_word) - 15.8
        self.reading_index['coleman-liau-level'] = round(level, 2)
        if self.verbose:
            print(f"Coleman-Liau Readability Index:".ljust(35, ' '), round(level, 2))

    def linsear_write(self):
        # Linsear Write Formula
        # https://en.wikipedia.org/wiki/Linsear_Write
        level = (self.total_ezwds + 3 * self.total_hdwds) / self.total_sents
        level = level/2 if level > 20 else (level/2 - 1)
        self.reading_index['linsear-level'] = round(level, 2)
        if self.verbose:
            print(f"Linsear Write Index:".ljust(35, ' '), round(level, 2))

    def smog_score2level(self, score):
        if score >= 0 and round(score, 2) < 2.999:
            level = 4
        elif score >= 3 and round(score, 2) < 6.999:
            level = 5
        elif score >= 7 and round(score, 2) < 12.999:
            level = 6
        elif score >= 13 and round(score, 2) < 20.999:
            level = 7
        elif score >= 21 and round(score, 2) < 42.999:
            level = 8
        elif score >= 43 and round(score, 2) < 56.999:
            level = 10
        elif score >= 57 and round(score, 2) < 72.999:
            level = 11
        elif score >= 73 and round(score, 2) < 90.999:
            level = 12
        elif score >= 91 and round(score, 2) < 110.999:
            level = 13
        elif score >= 111 and round(score, 2) < 132.999:
            level = 14
        elif score >= 133 and round(score, 2) < 156.999:
            level = 15
        elif score >= 157 and round(score, 2) < 182.999:
            level = 16
        elif score >= 183 and round(score, 2) < 210.999:
            level = 17
        else:
            level = 21
        return level

    def smog(self):
        # SMOG Index
        # https://en.wikipedia.org/wiki/SMOG
        score = 1.043 * math.sqrt(30 * self.total_hdwds / self.total_sents) + 3.1291
        self.reading_index['smog-score'] = round(score, 2)
        self.reading_index['smog-level'] = self.smog_score2level(score)
        if self.verbose:
            print(f"SMOG Readability score:".ljust(35, ' '), round(score, 2))
            print(f"SMOG Readability level:".ljust(35, ' '), self.reading_index['smog-level'])

    def forcast(self):
        # Forcast Readability Formula
        score = 20 - (150 * self.total_words_by_syll[0] / self.total_words) / 10
        self.reading_index['forcast'] = round(score, 2)
        if self.verbose:
            print(f"Forcast Readability score:".ljust(35, ' '), round(score, 2))

    def compute_indexes(self):
        self.flesch_reading_ease()
        self.automated_readability()
        self.gunning_fog()
        self.flesch_kincaid()
        self.coleman_liau()
        self.linsear_write()
        self.smog()
        self.forcast()
        self.levels = np.array([self.reading_index[x] for x in self.reading_index if x.endswith('-level')])
        self.level = self.levels.mean().round(2)
        if self.verbose:
            print(f"average level from {self.levels.size} results:".ljust(35, ' '), self.level)


if __name__ == "__main__":
    run = True
    if run:
        home_path = Path(os.getcwd())

        self = ReadingIndex(verbose=True)
        file = "./test_reading.txt"
        text = open(str(home_path.joinpath(file)), 'r+', encoding="utf-8").read()

        # compute statistics for reading indexes
        self.compute_text_statistics(text)

        # compute reading indexes
        self.compute_indexes()

