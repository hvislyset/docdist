import os
import re
import argparse

from math import sqrt, acos

parser = argparse.ArgumentParser(
    description='Compare two documents using cosine similarity')
parser.add_argument('-s', '--stop', default='stopwords.txt', type=str)
parser.add_argument('files', metavar='file.txt', type=str, nargs=2)
args = parser.parse_args()


def read_stopwords(f):
    """
    Return list of stopwords
    """
    stopwords = []

    if os.path.isfile(f):
        with open(f) as f:
            for line in f:
                words = line.replace('\n', '').lower()
                stopwords.append(words)
    else:
        raise FileNotFoundError

    return stopwords


def read_corpus(f, stopwords):
    """
    Return list of words
    """
    words = []

    if os.path.isfile(f):
        with open(f) as f:
            lines = " ".join(line.strip() for line in f).lower()
            for line in lines.split():
                line = re.sub(r'[^\w\s]', '', line)
                if line not in stopwords:
                    words.append(line)
    else:
        raise FileNotFoundError

    return words


def compute_frequency(words):
    """
    Return a frequency dict based on the words returned from the corpus
    """
    frequency = {}

    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1

    return frequency


def create_vector(frequency):
    """
    Return a vector based on the frequency dict
    """
    vector = []

    for key, value in frequency.items():
        vector.append(int(value))

    return vector


def normalise(vector):
    """
    Return the euclidean norm of a vector
    """
    norm = 0

    for value in vector:
        norm += pow(value, 2)

    return sqrt(norm)


def dot_product(vec1, vec2):
    """
    Return the dot product between two vectors
    """

    return sum(x * y for x, y in zip(vec1, vec2))


def distance(vec1, vec2):
    """
    Return the distance between two documents. Defined by cosine similarity measurement
    """
    return acos((dot_product(vec1, vec2)) /
                (normalise(vec1) * normalise(vec2)))


def compare_documents(stopwords, doc1, doc2):
    """
    Compare two documents
    """
    stopwords = read_stopwords(stopwords)
    words1 = read_corpus(doc1, stopwords)
    words2 = read_corpus(doc2, stopwords)

    frequency1 = compute_frequency(words1)
    frequency2 = compute_frequency(words2)

    vec1 = create_vector(frequency1)
    vec2 = create_vector(frequency2)

    dist = distance(vec1, vec2)

    return dist


def main():
    dist = compare_documents(args.stop, args.files[0], args.files[1])

    print(f"Distance score: {dist:.4f}")


if __name__ == "__main__":
    main()
