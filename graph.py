from collections import defaultdict, Counter
from typing import (
    Any, Collection, DefaultDict, Dict, List, Optional, Tuple
)

from fast_edit_distance import edit_distance
from tqdm.notebook import tqdm

DIRECTED = True

Words = Dict[str, bool]
Relation = Tuple[str, str]


class WordGraph:
    def __init__(self, words: Words, directed: bool = DIRECTED):
        self.words = words
        self.directed = directed
        self.graph, self.edges = self.create_graph()
        self.relation_map = self.get_relation_map()

    def __repr__(self) -> str:
        return f"WordGraph({len(self.words)} words, {len(self.edges)} unique edges)"

    def get_tuple(self, collection: Collection) -> Tuple:
        if self.directed:
            return tuple(collection)
        else:
            return tuple(sorted(collection))

    @staticmethod
    def relation_to_transition(relation: Relation) -> str:
        return " → ".join(relation)

    @staticmethod
    def boolean_pair_to_transition(b1: bool, b2: bool) -> str:
        return " → ".join(map(lambda b: "[+]" if b else "[-]", (b1, b2)))

    def get_edit_relation(self, word1: str, word2: str) -> Optional[Relation]:
        len1, len2 = len(word1), len(word2)
        for i in range(min(len1, len2)):
            if word1[i] != word2[i]:
                start = max(0, i - 1)
                end = min(len1, i + 2)
                if len1 == len2:
                    return self.get_tuple((word1[start:end], word2[start:end]))
                elif len1 < len2:
                    return self.get_tuple((word1[start:end - 1], word2[start:end]))
                else:
                    return self.get_tuple((word1[start:end], word2[start:end - 1]))

        start = min(len1, len2) - 1
        end = max(len1, len2)
        if len1 != len2:
            return self.get_tuple((word1[start:end], word2[start:end]))

        return None

    def create_graph(self) -> Tuple[DefaultDict[str, List[Tuple[str, Relation]]], Counter]:
        graph = defaultdict(list)
        edges = Counter()

        words = list(self.words.keys())
        lens = {word: len(word) for word in self.words}

        for i, word1 in enumerate(tqdm(words)):
            n = len(word1)

            for j in range(i + 1, len(words)):
                word2 = words[j]
                if abs(lens[word2] - n) > 1:
                    continue

                distance = edit_distance(word1, word2, max_ed=2)
                if distance == 1:
                    relation = self.get_edit_relation(word1, word2)

                    if relation:
                        graph[word1].append((word2, relation))
                        relation_prime = tuple(reversed(relation)) if self.directed else relation
                        graph[word2].append((word1, relation_prime))

                        edges[relation] += 1
                        if self.directed:
                            edges[relation_prime] += 1

        return graph, edges

    def get_relation_map(self, sort: bool = True) -> Dict[str, Dict[str, Any]]:
        relation_map = {}
        for relation in tqdm(self.edges):
            items = []
            for key, values in self.graph.items():
                for word, rel in values:
                    if rel == relation:
                        transition = self.boolean_pair_to_transition(self.words[key], self.words[word])
                        items.append(transition)

            symbol = self.relation_to_transition(relation)
            counts = self.edges[relation]
            relation_map[symbol] = {
                "transitions": {
                    key: value / (counts * (2 - self.directed))
                    for key, value in Counter(items).items()
                },
                "counts": counts
            }

        if sort:
            relation_map = dict(sorted(
                relation_map.items(), key=lambda x: x[1]["counts"], reverse=True
            ))

        return relation_map
