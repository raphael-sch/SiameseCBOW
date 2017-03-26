import numpy as np
import os


def write_w2v_format(embeddings, vocab, filename):
    """Write embeddings in word2vec format to file.

    Args:
        embeddings: weight matrix
        vocab: (dict) mapping of words to row indices in the weight matrix
        filename: path to the output file
    """
    print('write embeddings with shape {} to {}'.format(embeddings.shape, filename))
    dim = embeddings.shape[-1]
    f = open(filename, 'w')
    f.write(" ".join([str(len(vocab) - 1), str(dim)]))
    f.write("\n")

    # get word and weight matrix row index i
    for word, i in vocab.items():
        f.write(word)
        f.write(" ")
        # get vector a index i
        f.write(" ".join(map(str, list(embeddings[i, :]))))
        f.write("\n")
    f.close()


def read_w2v_format(filename, words=None):
    """Read a word2vec formatted file into an embedding representation.

    Args:
        filename: `string` path to file
        words: only read embeddings for these words

    Returns:
        `dict` word -> vector; dimension of embeddings
    """
    embeddings = dict()
    with open(filename, 'r') as f:
        dim = int(next(f).rstrip().split()[1])
        for line in f:
            line = line.rstrip().split(' ')
            token = line[0]
            if words is None or token in words:
                embeddings[token] = np.asarray(list(map(float, line[1:])))
    return embeddings, dim


def is_valid_file(parser, filename):
    """Check if file exists.

    Args:
        parser: `argparse.ArgumentParser`
        filename: `string` path to file

    """
    if not os.path.isfile(filename):
        parser.error("The file %s does not exist!" % filename)
    else:
        return filename
