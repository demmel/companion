"""
Research orchestrator for autonomous multi-cycle research.

Coordinates search → read → extract → think cycles to build knowledge graphs.
Simple sequential implementation - can be rewritten with different strategies.
"""

import logging
import requests
import re
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from agent.llm.router import LLM
from agent.llm.models import SupportedModel
from agent.llm.interface import Message

from .interfaces import IResearchOrchestrator, IFactExtractor, IKnowledgeGraph
from .knowledge_graph import SimpleHypergraph
from .config import ResearchConfig
from .web_search import search_duckduckgo

logger = logging.getLogger(__name__)

# Use Mistral 3.2 Q4 for all LLM calls
RESEARCH_MODEL = SupportedModel.MISTRAL_SMALL_3_2_Q4


class SequentialResearch(IResearchOrchestrator):
    """
    Simple sequential research orchestrator.

    Each cycle:
    1. Generate research questions
    2. Search for relevant sources
    3. Fetch and read top sources
    4. Extract facts from content
    5. Add facts to knowledge graph
    6. Generate follow-up questions based on findings

    Can be rewritten with parallel fetching, smarter source selection, etc.
    """

    def __init__(
        self, llm: LLM, fact_extractor: IFactExtractor, config: ResearchConfig
    ):
        self.llm = llm
        self.fact_extractor = fact_extractor
        self.config = config

    def research_topic(
        self, topic: str, depth: int = 3, initial_questions: Optional[List[str]] = None
    ) -> IKnowledgeGraph:
        """
        Conduct multi-cycle research on a topic.

        Args:
            topic: The topic to research
            depth: Number of research cycles
            initial_questions: Optional starting questions

        Returns:
            Knowledge graph built from research
        """
        logger.info(f"Starting research on topic: {topic} (depth={depth})")

        graph = SimpleHypergraph()
        questions = initial_questions or self._generate_initial_questions(topic)

        for cycle in range(depth):
            logger.info(f"Research cycle {cycle + 1}/{depth}")
            logger.info(f"Current questions: {questions}")

            # Search for sources
            search_results = self._search_for_sources(questions, topic)
            if not search_results:
                logger.warning(f"No search results in cycle {cycle + 1}, stopping")
                break

            # Fetch and extract from sources
            sources_to_fetch = search_results[: self.config.max_sources_per_cycle]
            new_facts = []
            for i, result in enumerate(sources_to_fetch):
                logger.info(
                    f"Fetching source {i + 1}/{len(sources_to_fetch)}: {result['url']}"
                )
                content = self._fetch_url_content(result["url"])

                if content:
                    # Extract facts
                    context = f"Research topic: {topic}\nSource: {result['title']}"
                    facts = self.fact_extractor.extract_facts(content, context)
                    new_facts.extend(facts)

                    # Add to graph with source tracking
                    for fact in facts:
                        graph.add_fact(fact, source_id=result["url"])

            logger.info(
                f"Cycle {cycle + 1}: extracted {len(new_facts)} facts (total: {len(graph)} facts)"
            )

            # Generate follow-up questions if not last cycle
            if cycle < depth - 1 and new_facts:
                questions = self._generate_followup_questions(topic, new_facts)
            else:
                break

        logger.info(
            f"Research complete. Built graph with {len(graph)} facts from {len(graph.get_all_entities())} entities"
        )
        return graph

    def _generate_initial_questions(self, topic: str) -> List[str]:
        """Generate initial research questions for a topic"""
        prompt = f"""Generate 3-5 specific research questions to explore the topic: "{topic}"

Questions should be:
- Specific and answerable with factual information
- Cover different aspects of the topic
- Guide productive web searches

Output format (one question per line):
1. [question]
2. [question]
...
"""

        try:
            response = self.llm.chat_complete(
                model=RESEARCH_MODEL,
                messages=[Message(role="user", content=prompt)],
                caller="generate_questions",
                temperature=0.7,
            )

            if not response:
                return [f"What is {topic}?", f"What are key facts about {topic}?"]

            # Parse numbered list
            questions = []
            for line in response.split("\n"):
                line = line.strip()
                if re.match(r"^\d+\.", line):
                    question = re.sub(r"^\d+\.\s*", "", line)
                    questions.append(question)

            return questions if questions else [f"What is {topic}?"]

        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return [f"What is {topic}?"]

    def _generate_followup_questions(
        self, topic: str, recent_facts: List[Any]
    ) -> List[str]:
        """Generate follow-up questions based on recently discovered facts"""
        # Summarize recent facts (limit if excessive to fit in context)
        max_facts = self.config.max_facts_for_followup
        facts_to_summarize = (
            recent_facts[:max_facts] if len(recent_facts) > max_facts else recent_facts
        )

        fact_summary = "\n".join(
            [
                f"- {f.predicate}: {', '.join(f.entities.values())}"
                for f in facts_to_summarize
            ]
        )

        if len(recent_facts) > max_facts:
            fact_summary += f"\n... and {len(recent_facts) - max_facts} more facts"

        prompt = f"""Based on research into "{topic}", I've discovered these facts:

{fact_summary}

Generate 2-3 follow-up questions to deepen understanding. Focus on:
- Unexplored connections or relationships
- Missing context or details
- Related concepts worth investigating

Output format (one question per line):
1. [question]
2. [question]
..."""

        try:
            response = self.llm.chat_complete(
                model=RESEARCH_MODEL,
                messages=[Message(role="user", content=prompt)],
                caller="generate_followup",
                temperature=0.7,
            )

            if not response:
                return [f"What else about {topic}?"]

            # Parse numbered list
            questions = []
            for line in response.split("\n"):
                line = line.strip()
                if re.match(r"^\d+\.", line):
                    question = re.sub(r"^\d+\.\s*", "", line)
                    questions.append(question)

            return questions if questions else [f"What else about {topic}?"]

        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return [f"What else about {topic}?"]

    def _search_for_sources(
        self, questions: List[str], topic: str
    ) -> List[Dict[str, str]]:
        """Search web for sources relevant to questions"""
        # Combine questions into search query
        # Include all questions but limit total length to reasonable search query size
        MAX_QUERY_LENGTH = 300
        query_parts = [topic]

        for q in questions:
            test_query = " ".join(query_parts + [q])
            if len(test_query) <= MAX_QUERY_LENGTH:
                query_parts.append(q)
            else:
                break  # Stop adding questions if we'd exceed limit

        query = " ".join(query_parts)

        logger.info(f"Searching: {query}")

        # Use shared search function
        results = search_duckduckgo(query, max_results=10)
        logger.info(f"Found {len(results)} search results")
        return results

    def _fetch_url_content(self, url: str) -> Optional[str]:
        """Fetch and extract text content from URL"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(
                url, headers=headers, timeout=10, allow_redirects=True
            )
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Convert to markdown
            markdown_content = md(str(soup), heading_style="ATX")

            # Clean up excessive whitespace
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
            markdown_content = markdown_content.strip()

            # Don't truncate here - let the fact extractor handle chunking if needed
            return markdown_content

        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return None
