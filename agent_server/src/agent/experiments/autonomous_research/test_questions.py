"""
Test questions dataset for evaluating retrieval quality.

Provides ground truth for measuring whether the system can answer
questions about researched topics.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class TopicTestQuestions:
    """Test questions and expected entities/predicates for a topic"""

    topic: str
    questions: List[str]
    expected_entities: List[str]  # Entities we expect to find
    expected_predicates: List[str]  # Relationship types we expect

    def entity_recall(self, found_entities: List[str]) -> float:
        """Calculate recall of expected entities"""
        found_set = set(e.lower() for e in found_entities)
        expected_set = set(e.lower() for e in self.expected_entities)

        if not expected_set:
            return 1.0

        found_count = sum(1 for e in expected_set if any(e in f for f in found_set))
        return found_count / len(expected_set)

    def predicate_recall(self, found_predicates: List[str]) -> float:
        """Calculate recall of expected predicates"""
        found_set = set(p.lower() for p in found_predicates)
        expected_set = set(p.lower() for p in self.expected_predicates)

        if not expected_set:
            return 1.0

        found_count = sum(1 for p in expected_set if any(p in f for f in found_set))
        return found_count / len(expected_set)


# Test questions for common topics
TEST_QUESTIONS: Dict[str, TopicTestQuestions] = {
    "Byzantine Empire": TopicTestQuestions(
        topic="Byzantine Empire",
        questions=[
            "What was the capital of the Byzantine Empire?",
            "Who were famous Byzantine emperors?",
            "What were Byzantine trade relationships?",
            "What was Byzantine art and culture like?",
            "When did the Byzantine Empire fall?",
        ],
        expected_entities=[
            "Constantinople",
            "Justinian",
            "Theodora",
            "Venice",
            "Ottoman",
            "Hagia Sophia",
        ],
        expected_predicates=[
            "capital",
            "ruled_by",
            "traded_with",
            "conquered_by",
            "known_for",
            "located_at",
        ],
    ),
    "Quantum Computing": TopicTestQuestions(
        topic="Quantum Computing",
        questions=[
            "What is a qubit?",
            "How does quantum superposition work?",
            "What are quantum computing applications?",
            "Who are leading quantum computing companies?",
            "What is quantum entanglement?",
        ],
        expected_entities=[
            "qubit",
            "superposition",
            "entanglement",
            "IBM",
            "Google",
            "algorithm",
        ],
        expected_predicates=[
            "defined_as",
            "uses",
            "enables",
            "developed_by",
            "based_on",
            "applies_to",
        ],
    ),
    "Coffee": TopicTestQuestions(
        topic="Coffee",
        questions=[
            "Where is coffee grown?",
            "What are coffee brewing methods?",
            "What is espresso?",
            "Where did coffee originate?",
            "What affects coffee flavor?",
        ],
        expected_entities=[
            "Ethiopia",
            "espresso",
            "arabica",
            "robusta",
            "brewing",
            "roasting",
        ],
        expected_predicates=[
            "grown_in",
            "method",
            "originated_from",
            "type_of",
            "affects",
            "produced_by",
        ],
    ),
    "Machine Learning": TopicTestQuestions(
        topic="Machine Learning",
        questions=[
            "What is supervised learning?",
            "What are neural networks?",
            "Who developed deep learning?",
            "What are ML applications?",
            "What is training data?",
        ],
        expected_entities=[
            "neural network",
            "supervised learning",
            "deep learning",
            "training data",
            "algorithm",
            "model",
        ],
        expected_predicates=[
            "type_of",
            "uses",
            "requires",
            "trained_on",
            "developed_by",
            "applied_to",
        ],
    ),
    "Coffee brewing": TopicTestQuestions(
        topic="Coffee brewing",
        questions=[
            "What are pour-over methods?",
            "How does espresso extraction work?",
            "What is cold brew?",
            "What affects extraction?",
            "What equipment is needed?",
        ],
        expected_entities=[
            "pour over",
            "espresso",
            "cold brew",
            "grind size",
            "temperature",
            "extraction",
        ],
        expected_predicates=[
            "method",
            "requires",
            "affects",
            "produces",
            "involves",
            "uses",
        ],
    ),
}


def get_test_questions(topic: str) -> TopicTestQuestions:
    """Get test questions for a topic"""
    # Exact match
    if topic in TEST_QUESTIONS:
        return TEST_QUESTIONS[topic]

    # Fuzzy match (case-insensitive, partial)
    topic_lower = topic.lower()
    for key, value in TEST_QUESTIONS.items():
        if topic_lower in key.lower() or key.lower() in topic_lower:
            return value

    # Return generic questions
    return TopicTestQuestions(
        topic=topic,
        questions=[
            f"What is {topic}?",
            f"What are key facts about {topic}?",
            f"What is the history of {topic}?",
        ],
        expected_entities=[],
        expected_predicates=[],
    )


def get_all_topics() -> List[str]:
    """Get all topics with test questions"""
    return list(TEST_QUESTIONS.keys())
