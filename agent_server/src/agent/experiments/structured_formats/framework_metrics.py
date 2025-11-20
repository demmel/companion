"""
Metrics calculator for structured format experiments using the framework.
"""

from typing import Optional, Dict, List
import logging
from pydantic import BaseModel

from agent.experiments.framework import MetricsCalculator
from agent.llm import LLM, SupportedModel

from .evaluation import evaluate_correctness
from .semantic_evaluation import semantic_evaluate

logger = logging.getLogger(__name__)


class StructuredFormatMetricsCalculator(MetricsCalculator):
    """
    Metrics calculator for structured output format experiments.

    Supports different evaluation modes:
    - Strict: Exact match evaluation
    - Flexible: Flexible evaluation with hardcoded synonyms
    - Semantic: Semantic similarity via embeddings
    """

    def __init__(
        self,
        llm: LLM,
        model: SupportedModel,
        use_semantic_eval: bool,
    ):
        """
        Initialize metrics calculator.

        Args:
            use_semantic_eval: Use semantic evaluation (overrides flexible)
            llm: LLM instance for comparative metrics (required for richness)
            model: Model to use for LLM-as-judge
        """
        self.use_semantic_eval = use_semantic_eval
        self.llm = llm
        self.model = model

    def calculate(
        self, output: BaseModel, expected: Optional[BaseModel]
    ) -> dict[str, float]:
        """
        Calculate per-run metrics by comparing output to expected.

        Args:
            output: The output from test case execution
            expected: The expected output (may be None)

        Returns:
            Dictionary with precision, recall, and f1 metrics
        """
        if expected is None:
            return {}

        # Use semantic, flexible, or strict evaluation based on configuration
        if self.use_semantic_eval:
            precision, recall, f1 = semantic_evaluate(output, expected)
        else:
            precision, recall, f1 = evaluate_correctness(output, expected)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def calculate_comparative(
        self,
        test_case_name: str,
        variant_outputs: Dict[str, List[BaseModel]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate comparative metrics across variants using LLM-as-judge.

        Args:
            test_case_name: Name of the test case being compared
            variant_outputs: Dict mapping variant_name -> list of outputs from all runs

        Returns:
            Dictionary mapping metric_name -> variant_name -> score
            Example: {"richness": {"json": 0.9, "xml": 0.7, "yaml": 0.8, "sexp": 0.6}}
        """
        if self.llm is None or self.model is None:
            logger.warning(
                "LLM not configured for comparative metrics, skipping richness calculation"
            )
            return {}

        # Calculate richness using LLM-as-judge
        richness_scores = self._calculate_richness_comparative(
            test_case_name, variant_outputs
        )

        return {"richness": richness_scores}

    def _calculate_richness_comparative(
        self,
        test_case_name: str,
        variant_outputs: Dict[str, List[BaseModel]],
    ) -> Dict[str, float]:
        """
        Calculate richness scores using LLM-as-judge.

        Richness is a qualitative measure of how detailed, complete, and informative
        the output is. We use an LLM to compare outputs across variants.

        Args:
            test_case_name: Name of the test case
            variant_outputs: Dict mapping variant_name -> list of outputs

        Returns:
            Dict mapping variant_name -> richness score (0.0-1.0)
        """
        # Sample one output from each variant for comparison
        # (Using first successful output for consistency)
        samples = {}
        for variant_name, outputs in variant_outputs.items():
            if outputs:
                samples[variant_name] = outputs[0]

        if len(samples) < 2:
            # Need at least 2 variants to compare
            return {}

        # Build prompt for LLM-as-judge
        prompt = self._build_richness_comparison_prompt(test_case_name, samples)

        try:
            # Call LLM
            response = self.llm.generate_complete(
                model=self.model,
                prompt=prompt,
                num_predict=1024,
                caller="richness_comparison",
                temperature=0.1,
            )

            # Parse LLM response to extract scores
            scores = self._parse_richness_scores(response, list(samples.keys()))

            logger.info(f"Richness scores for {test_case_name}: {scores}")
            return scores

        except Exception as e:
            logger.error(f"Error calculating richness for {test_case_name}: {e}")
            return {}

    def _build_richness_comparison_prompt(
        self,
        test_case_name: str,
        samples: Dict[str, BaseModel],
    ) -> str:
        """
        Build prompt for LLM to compare richness across variants.

        Args:
            test_case_name: Name of the test case
            samples: Dict mapping variant_name -> sample output

        Returns:
            Prompt string
        """
        prompt_lines = [
            "You are evaluating the richness (detail, completeness, informativeness) of different structured output formats.",
            f"\nTest case: {test_case_name}",
            "\nCompare the following outputs and assign a richness score from 0.0 (least rich) to 1.0 (most rich) to each format.",
            "Consider:",
            "- How much detail is preserved",
            "- How complete the information is",
            "- How easy it is to understand the structure",
            "- How much context is provided",
            "\nOutputs to compare:\n",
        ]

        # Add each variant's output
        for variant_name, output in samples.items():
            output_json = output.model_dump_json(indent=2)
            prompt_lines.append(f"\n{variant_name.upper()}:")
            prompt_lines.append(f"```\n{output_json}\n```")

        prompt_lines.append("\nProvide your analysis in this exact format:")
        prompt_lines.append("SCORES:")
        for variant_name in samples.keys():
            prompt_lines.append(f"{variant_name}: <score>")

        prompt_lines.append(
            "\nProvide ONLY the scores in the format shown above, with one score per line."
        )

        return "\n".join(prompt_lines)

    def _parse_richness_scores(
        self,
        response: str,
        variant_names: List[str],
    ) -> Dict[str, float]:
        """
        Parse LLM response to extract richness scores.

        Args:
            response: LLM response text
            variant_names: List of variant names to extract scores for

        Returns:
            Dict mapping variant_name -> score
        """
        scores = {}

        # Look for "SCORES:" section
        lines = response.split("\n")
        in_scores_section = False

        for line in lines:
            line = line.strip()

            if "SCORES:" in line.upper():
                in_scores_section = True
                continue

            if in_scores_section:
                # Try to parse "variant_name: score"
                for variant_name in variant_names:
                    if variant_name.lower() in line.lower():
                        # Extract score (look for number between 0 and 1)
                        parts = line.split(":")
                        if len(parts) >= 2:
                            score_str = parts[1].strip()
                            try:
                                score = float(score_str)
                                if 0.0 <= score <= 1.0:
                                    scores[variant_name] = score
                            except ValueError:
                                logger.warning(
                                    f"Could not parse score from line: {line}"
                                )

        # Fallback: if we couldn't parse scores, assign equal scores
        if not scores:
            logger.warning(
                "Could not parse richness scores from LLM response, using equal scores"
            )
            for variant_name in variant_names:
                scores[variant_name] = 0.5

        return scores
