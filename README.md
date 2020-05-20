# Active Sampling Pipeline

Active sampling is an active learning strategy to construct diverse, balanced training data sets for training text classifiers.
These combine text processing innovations like real-valued document embeddings with traditional active learning pipelines.

This pipeline operates on (i) a training data set, (ii) an unlabeled pool of comments, (iii) a text classifier. And iteratively augments the training data set using the classifier and then improves the classifier using the newly acquired training data set.

## Table of Contents

* [Background](#background)
* [Dependencies](#dependencies)
* [Execution](#execution)

### Background


Many classification problems involve predicting classes (labels) where the dominant class forms an overwhelmingly significant portion of the dataset. A high accuracy or low categorical cross-entropy loss can be achieved by simply predicting the dominant label. Machine learning practitioners advise using better metrics to accurately capture classifier performance and a variety of approaches to improve performance for the minority label. We conceived an idea and applied it, in the setting of active learning where a model is trained over a series of iterations with a human-in-the-loop setup. We considered how to address label imbalance with active learning approaches across modalities.

Active learning approaches typically combine several components: (1) a sampling strategy to select examples to label for the next training round, (2) metrics that capture the end goal of the learning system, and (3) strategies that prevent biases that can influence the sampling approaches. A further choice is how many examples to label in a given round, which is often determined by the problem setting. For example, it may depend on access to annotators.

The following are a few popular sampling strategies.

Random Sampling: Points are sampled from the dataset at random. This type of sampling retrieves points that have the same underlying prior distribution as the data. The class imbalance in the data will influence the sampling and the sampled points are likely to be from the majority class.
Uncertainty Sampling: This is typically utilized to address those data points where the classifier confidence is low. Standard strategies to choose these points exploit classifier confidence. Low confidence predictions (for instance probability of assigned label) are then labeled and fed in for a next round of training.
Minority Class Sampling: The system only randomly samples points from the minority class until a reasonable balance is achieved.
One-sided Certainty Sampling: The classifier picks points that are (1) from the minority class and (2) predicted with high confidence and then labels them. The end goal of this sampling strategy is to obtain minority class data points, and (ii) improve the classification precision (fraction of true positives from the predicted positives) of the classifier.

Typical active learning methods combine multiple sampling strategies either in separate rounds or in a sequence of rounds. With a dataset of text comments, described in a prior report, we employed a round of random, uncertainty, and finally certainty sampling to address drastic class imbalance. The first random sampling round was used to build a small dataset and train a simple support-vector machine (SVM) text classifier. Then, uncertainty sampling was utilized to address those documents predicted with low confidence, and finally minority-class certainty sampling was used to sample a large set of documents from the minority class. In comparison to random sampling that yields nearly a 90-10 split between the labels, the resulting dataset was almost evenly split between two labels. 

In addition to the above sampling strategies, newer methods leverage large-scale pretrained deep-learning models. Examples of such models include BERT, FastText, ElMo for text. These allow an end system to obtain real-valued vectors (embeddings) for a document. These embeddings capture a variety of syntactic and semantic natural language features. Representing data points as embeddings allows us to measure similarities and incorporate a form of near-neighbor sampling that captures some desired properties. This can be used, for example, to sample text that is similar to an example document. We used a popular document embedding model to sample social media posts similar to examples written by annotators (i.e. the examples are not part of the original dataset). 

Embeddings from large-scale machine learning models are popularly employed by a variety of other modalities as well. ResNet, VGG models are used to extract real-valued vectors for an input image and treat this vector as a feature-vector for additional nearest neighbor sampling tasks. 

The work we performed successfully saw handling of an imbalanced training set. For future work, we propose utilizing embeddings from beyond text tasks for addressing a variety of class imbalance problems. Nearest neighbor sampling can be utilized in ways that allow annotators to express a variety of concepts. For instance, when sampling items from a large graph, real-valued embeddings for nodes can allow annotators to sample a large and diverse set of examples (even in cases with extreme skew like power-law distribution).


### Dependencies

You can install all dependencies using:

```
pip install -r requirements.txt
```

### Execution

An active sampling pipeline is a sequence of steps where (i) a text classifier is trained on an acquired data set, (ii) the classifier from the previous step is utilized to find unlabeled documents that annotators label and augment the existing data set with.

The steps in this pipline are:

* Train an initial classifier on a corpus of text documents.

```
python train_classifier.py /path/to/training/dataset.txt /path/to/save/model.1.joblib
```

An example file with the desired text format is present in [example_training_set.txt](example_training_set.txt)

* Conduct a round of minority-class certainty sampling:

```
python label_file.py /path/to/unlabeled_corpus.txt /path/to/save/model.1.joblib /path/to/save/results.ndjson
```

Certainty sampling retrieves the most confident (highest likelyhood) minority class predictions from a corpus. The above command saves at `/path/to/save/results.ndjson`, a file with the text, and the prediction probability for each line in `/path/to/unlabeled_corpus.txt`. The resulting file has 1 json object per line.

* One the above file is renamed, and the dataset augmented with the samples from the above line, we run a training step to update the classifier:

```
python train_classifier.py /path/to/training/dataset+certainty.txt /path/to/save/model.2.joblib
```

Next, 

```
python label_file_uncertainty.py /path/to/unlabeled_corpus.txt /path/to/save/model.2.joblib /path/to/save/results.ndjson
```

After training and augmenting, we can also conduct a random sampling round:


```
python sample_comments.py /path/to/unlabeled_corpus.txt /path/to/save/results.ndjson
```

The final data set is the resulting training data set. A final classifier can be trained on this data set using the `train_classifier.py` script (see above for usage).

The training data set and ndjson formats are incompatible. Thus a utility script is provided: [generate_svm_data.py](generate_svm_data.py) to consume the `ndjson` and produce a text file usable by the text classifier.
