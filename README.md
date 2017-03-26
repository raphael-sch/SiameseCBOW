## Description
This repository is a Python implementation of Siamese CBOW [1]. It is used to learn word embeddings optimized for sentence representation. Like word2vec the embeddings are learned by providing positive and negative context examples. The difference is, that these context examples are sentences and not single words, although a sentence is represented as the average of its word vectors.<br \>
The implementation can easily be extended to support other input corpora than Toronto BookCorpus [2][3].
---
![alt text](https://raw.githubusercontent.com/raphael-sch/TensorflowSiameseCBOW/master/images/model.png "Diagram of the model")
<br /> 
Siamese CBOW network architecture [1]
---
<br />
<br />

[1] Kenter et al., 2016, Siamese CBOW: Optimizing Word Embeddings for Sentence Representations, https://arxiv.org/abs/1606.04640 <br />
[2] Download Toronto BookCorpus, http://yknzhu.wixsite.com/mbweb
[3] Zhu et al., 2015, Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books, https://arxiv.org/abs/1506.06724 <br /> 

