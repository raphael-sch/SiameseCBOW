from random import choice
from feeder import Feeder


class SimpleFeeder(Feeder):
    """Input file has to be in the format:

    one sentence per line
    word tokenized and separated by one single space

    """

    def __init__(self, filename, min_count=5, neg_sampling=2, available_words=None, verbose=1):
        """

        Args:
            filename: path to the kisti data file
            min_count: dismiss all words with count < min_count
            neg_sampling: number of negative titles
            available_words: only consider words in this iterable
            verbose: verbosity of output
        """
        super(SimpleFeeder, self).__init__(min_count=min_count, pos_sampling=2, neg_sampling=neg_sampling, verbose=verbose)
        self.filename = filename
        self.sentences = list()

        self._read_data(available_words)

    def _read_data(self, available_words=None):
        """ Reads the tokenized sentences

        Args:
            available_words: only consider words in this iterable

        Returns:

        """

        # iterate trough sentences and count words
        word_count = dict()
        if self.verbose > 0:
            print('count words...')
        with open(self.filename) as f:
            for i, line in enumerate(f):
                line = line.rstrip('\n')
                sentence = line.split(' ')
                for word in sentence:
                    word_count[word] = word_count.get(word, 0) + 1

        # remove words which are not available
        if available_words is not None:
            for word in word_count.keys():
                if word not in available_words:
                    word_count[word] = -1

        if self.verbose > 0:
            print('read sentences from filename: {}'.format(self.filename))

        # iterate a second time and store sentences as word indices
        with open(self.filename) as f:
            for i, line in enumerate(f):
                line = line.rstrip('\n')
                sentence = line.split(' ')
                for word in sentence:
                    if word_count[word] >= self.min_count:
                        self.vocab[word] = self.vocab.get(word, len(self.vocab))
                sentence = [self.vocab[word] for word in sentence if word in self.vocab]
                # don't add sentence if no words are left after removing infrequent words
                if len(sentence) > 0:
                    self.sentences.append(sentence)

        if self.verbose > 0:
            print('found {} sentences'.format(len(self.sentences)))

    def base_sentence_generator(self):
        """Yields every sentence in the corpus once. Provides information to choose the compare sentences.

        Returns: (list word word ids, sentence id)

        """
        for sentence_id in range(1, len(self.sentences) - 1):
            yield self.sentences[sentence_id], sentence_id

    def get_pos_sentences(self, info=None):
        """ Returns the preceding and following sentence as positive compare sentences.

        Args:
            info: (sentence id)

        Returns: list of list of word indices

        """
        if info is None:
            raise ValueError('Need sentence_id of base sentence as info to retrieve positive examples')
        pos_sentence_1 = self.sentences[info - 1]
        pos_sentence_2 = self.sentences[info + 1]
        return [pos_sentence_1, pos_sentence_2]

    def get_neg_sentences(self, info=None):
        """Get negative sentence by randomly drawing one from the corpus.

        Args:
            info: (sentence id)

        Returns: list of list of word ids

        """
        neg_sentences = list()
        for _ in range(self.neg_sampling):
            sentence_id = None
            while sentence_id is None:
                sentence_id = choice(range(len(self.sentences)))
                if info is not None and sentence_id == info:
                    sentence_id = None
            neg_sentences.append(self.sentences[sentence_id])
        return neg_sentences
