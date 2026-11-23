"""Dependency-free CIDEr-D-style reward with corpus document frequencies.

This is used only as the policy-gradient training reward.  Reported benchmark
numbers still come from the official COCO PTB tokenizer and CIDEr evaluator.
"""
from collections import Counter, defaultdict
import math
import re

import torch


TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def words(text):
    return TOKEN.findall(text.lower())


def counts(text):
    tokens = words(text)
    result = []
    for order in range(1, 5):
        result.append(Counter(
            tuple(tokens[index:index + order])
            for index in range(max(0, len(tokens) - order + 1))))
    return result


class CiderReward:
    def __init__(self, corpus_references, sigma=6.0):
        self.sigma = sigma
        self.document_frequency = [defaultdict(float) for _ in range(4)]
        for references in corpus_references:
            per_document = [set() for _ in range(4)]
            for reference in references:
                encoded = counts(reference)
                for order in range(4):
                    per_document[order].update(encoded[order])
            for order in range(4):
                for ngram in per_document[order]:
                    self.document_frequency[order][ngram] += 1
        self.log_documents = math.log(max(1, len(corpus_references)))

    def vector(self, encoded):
        vectors = []
        norms = []
        for order, terms in enumerate(encoded):
            vector = {}
            squared = 0.0
            for ngram, frequency in terms.items():
                document_frequency = max(
                    1.0, self.document_frequency[order].get(ngram, 0.0))
                value = frequency * (
                    self.log_documents - math.log(document_frequency))
                vector[ngram] = value
                squared += value * value
            vectors.append(vector)
            norms.append(math.sqrt(squared))
        length = sum(encoded[0].values())
        return vectors, norms, length

    def similarity(self, candidate, reference):
        candidate_vectors, candidate_norms, candidate_length = self.vector(candidate)
        reference_vectors, reference_norms, reference_length = self.vector(reference)
        penalty = math.exp(
            -((candidate_length - reference_length) ** 2)
            / (2 * self.sigma * self.sigma))
        score = 0.0
        for order in range(4):
            denominator = candidate_norms[order] * reference_norms[order]
            if denominator == 0:
                continue
            overlap = 0.0
            for ngram, candidate_value in candidate_vectors[order].items():
                reference_value = reference_vectors[order].get(ngram, 0.0)
                # CIDEr-D clips term-frequency contribution while preserving
                # the independently computed TF-IDF weights.
                overlap += min(candidate_value, reference_value) * reference_value
            score += overlap / denominator * penalty
        return score / 4.0

    def score(self, candidates, references, device=None):
        result = []
        for candidate, candidate_references in zip(candidates, references):
            encoded = counts(candidate)
            values = [self.similarity(encoded, counts(reference))
                      for reference in candidate_references]
            result.append(10.0 * sum(values) / max(1, len(values)))
        return torch.tensor(result, dtype=torch.float32, device=device)
