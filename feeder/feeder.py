import warnings


class Feeder(object):
    """Subclass this to write your own corpus feeder

    """

    def __init__(self, min_count=1, pos_sampling=1, neg_sampling=2, verbose=0):
        if min_count < 1:
            warnings.warn('min_count must be greater than 0; set to default 1')
            min_count = 1
        if neg_sampling < 1:
            warnings.warn('Need at least one negative sample; neg_sampling set to default 2')
            neg_sampling = 2
        if pos_sampling < 1:
            warnings.warn('Need at least one positive sample; pos_sampling set to default 1')
            pos_sampling = 1
        self.vocab = dict()
        self.min_count = min_count
        self.pos_sampling = pos_sampling
        self.neg_sampling = neg_sampling
        self.verbose = verbose

    def base_sentence_generator(self):
        """Yield base sentence until exhausted.

        Returns: `tuple` (list of word indices, info to be passed)

        """
        pass

    def get_pos_sentences(self, info=None):
        """ Generate n positive sentence partners. n = pos_sampling

        Args:
            info: passed from base_sentence_generator. Can be used to find positive sentence.

        Returns: list of list of word indices

        """
        pass

    def get_neg_sentences(self, info=None):
        """ Generate n negative sentence partners. n = neg_sampling

        Args:
            info: passed from base_sentence_generator. Can be used to find negative sentence.

        Returns: list of list of word indices

        """
        pass
