def get_instances(feeder, batch_size=128):
    """Generate sparse input for the model inputs.

    Args:
        feeder: `Feeder` holds settings and data of current corpus
        batch_size: size of the batch to be returned

    Returns: indices, values, shape for the three model inputs

    """
    pos_sampling = feeder.pos_sampling
    neg_sampling = feeder.neg_sampling

    base_sentences_batch = list()
    pos_sentences_batches = [list() for _ in range(pos_sampling)]
    neg_sentences_batches = [list() for _ in range(neg_sampling)]
    for base_sentence, info in feeder.base_sentence_generator():
        base_sentences_batch.append(base_sentence)

        # get positive samples
        pos_sentences = feeder.get_pos_sentences(info)
        for i in range(pos_sampling):
            pos_sentences_batches[i].append(pos_sentences[i])

        # get negative samples
        neg_sentences = feeder.get_neg_sentences(info)
        for i in range(neg_sampling):
            neg_sentences_batches[i].append(neg_sentences[i])

        # yield inputs if batch size is reached
        if len(base_sentences_batch) == batch_size:
            base_input = get_sparse_input(base_sentences_batch)
            pos_inputs = [get_sparse_input(pos_sentences_batches[i]) for i in range(pos_sampling)]
            neg_inputs = [get_sparse_input(neg_sentences_batches[i]) for i in range(neg_sampling)]

            yield base_input, pos_inputs, neg_inputs
            base_sentences_batch = list()
            pos_sentences_batches = [list() for _ in range(pos_sampling)]
            neg_sentences_batches = [list() for _ in range(neg_sampling)]


def get_sparse_input(sentences):
    """Generate sparse representation of the given sentences.

    Args:
        sentences: list of list of word ids

    Returns:
        indices: list of coordinates where value is not 0;
        values: list of values;
        shape: dense shape
    """
    indices = list()
    values = list()
    max_len = 0

    for d1, sentence in enumerate(sentences):
        for d2, word_id in enumerate(sentence):
            indices.append((d1, d2))
            values.append(word_id)
            max_len = max(max_len, len(sentence))
    shape = (len(sentences), max_len)
    return indices, values, shape
