import argparse
from utils import is_valid_file
import random
random.seed(1337)


def run():
    """Parse command line arguments and start workflow.

    """
    parser = argparse.ArgumentParser(description='Run training of Siamese CBOW')

    parser.add_argument('corpus_name', type=str, choices=['simple'],
                        help='Name of the corpus to use for training. simple: one tokenized sentence per line')

    parser.add_argument('corpus_file', type=lambda x: is_valid_file(parser, x),
                        help='Path to the data file of the chosen corpus.')

    parser.add_argument('output_file', type=str,
                        help='Path to file where trained embeddings should be written to.'
                             'All command line arguments can be used in the name.'
                             'Example: siamese_kisti_dim{dim}_ep{epochs}_neg{neg_sampling}.w2v')

    parser.add_argument('-dim', nargs='?', type=int, default=100,
                        help='Dimension of the trained embeddings.')

    parser.add_argument('-min_count', nargs='?', type=int, default=1,
                        help='Learn only embeddings for words with count >= min_count.')

    parser.add_argument('-neg_sampling', nargs='?', type=int, default=2,
                        help='Number of negative sentences sampled.')

    parser.add_argument('-epochs', nargs='?', type=int, default=500,
                        help='Number of training epochs.')

    parser.add_argument('-batch_size', nargs='?', type=int, default=128,
                        help='Size of training batches.')

    parser.add_argument('-verbose', nargs='?', type=int, choices=[0, 1, 2], default=2,
                        help='Verbosity of output.')

    parser.add_argument('-init_weights', type=lambda x: is_valid_file(parser, x),
                        help='Read the weights from this word2vec formatted file to initialize embeddings. Use this to '
                             'resume previous training or optimize embeddings from other sources')

    args = parser.parse_args()
    args = vars(args)
    if args['verbose'] > 0:
        print('got command line arguments:')
        print(args)
    workflow(**args)


def workflow(corpus_name, corpus_file, output_file, dim, min_count, neg_sampling, epochs, batch_size, verbose,
             init_weights):
    """Prepare data to run the training.

    Args:
        corpus_name: Name of the corpus to use for training. tbc or kisti
        corpus_file: Path to the data file of the chosen corpus.
        output_file: Path to file where trained embeddings should be written to.
        dim: Dimension of the trained embeddings.
        min_count: Learn only embeddings for words with count >= min_count.
        neg_sampling: Number of negative sentences sampled.
        epochs: Number of training epochs.
        batch_size: Size of training batches.
        verbose: Verbosity of output.
        init_weights: Read the weights from this word2vec formatted file to initialize embeddings.

    Returns:
    runs the training and writes embeddings to output_file
    """

    # import bigger packages after argparse to quickly provide help messages through command line
    from model import train
    from utils import read_w2v_format
    # load chosen feeder
    if corpus_name == 'simple':
        from feeder.simple_corpus import SimpleFeeder as Feeder

    # load embeddings if init file is provided
    init_embeddings = None
    available_words = None
    if init_weights is not None:
        init_embeddings_file = init_weights
        init_embeddings, _ = read_w2v_format(init_embeddings_file)
        available_words = init_embeddings.keys()

    # read corpus data into feeder object
    feeder = Feeder(corpus_file, min_count=min_count, neg_sampling=neg_sampling, available_words=available_words,
                    verbose=verbose)

    train(feeder, output_file, dim=dim, epochs=epochs, batch_size=batch_size, verbose=verbose,
          init_embeddings=init_embeddings)


if __name__ == '__main__':
    run()
