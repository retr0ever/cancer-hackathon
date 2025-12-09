"""
Tumour Grade Estimation Module for PathoLens

Estimates histological grade based on aggregated features from
cancer classification and nuclei segmentation analysis.

IMPORTANT: This module is for research demonstration only and
should not be used for clinical diagnosis.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy.stats import entropy


@dataclass
class GradeEstimate:
    """Tumour grade probability estimates."""
    grade_1: float  # Well-differentiated
    grade_2: float  # Moderately differentiated
    grade_3: float  # Poorly differentiated
    confidence: float
    primary_grade: int
    features_used: List[str]


@dataclass
class GradingFeatures:
    """Features used for tumour grading."""
    mean_cancer_probability: float
    max_cancer_probability: float
    cancer_area_fraction: float
    nuclear_pleomorphism_score: float  # Based on size variation
    nuclear_atypia_score: float  # Based on shape irregularity
    mitotic_activity_estimate: float  # Based on nuclear density
    texture_heterogeneity: float  # Patch-level entropy
    malignant_cell_density: float


class TumourGrader:
    """
    Estimates tumour grade using aggregated histopathological features.

    Grading is based on the Nottingham Grading System concepts:
    - Tubule formation (approximated by tissue organisation)
    - Nuclear pleomorphism (nuclear size/shape variation)
    - Mitotic count (approximated by nuclear density patterns)

    Note: This is a simplified demonstration model.
    """

    def __init__(self, use_softmax: bool = True):
        """
        Initialise the tumour grader.

        Args:
            use_softmax: Whether to normalise outputs with softmax
        """
        self.use_softmax = use_softmax

        # Weights for grade estimation (learned or rule-based)
        # These are demonstration values
        self.grade_weights = {
            'grade_1': {
                'cancer_prob': -0.3,
                'pleomorphism': -0.4,
                'atypia': -0.3,
                'mitotic': -0.2,
                'heterogeneity': -0.2,
                'bias': 0.8
            },
            'grade_2': {
                'cancer_prob': 0.1,
                'pleomorphism': 0.1,
                'atypia': 0.1,
                'mitotic': 0.1,
                'heterogeneity': 0.1,
                'bias': 0.3
            },
            'grade_3': {
                'cancer_prob': 0.4,
                'pleomorphism': 0.5,
                'atypia': 0.4,
                'mitotic': 0.3,
                'heterogeneity': 0.3,
                'bias': -0.5
            }
        }

    def extract_grading_features(
        self,
        heatmap_stats: Dict,
        nuclei_stats: Dict,
        patch_probabilities: List[float]
    ) -> GradingFeatures:
        """
        Extract features relevant to tumour grading.

        Args:
            heatmap_stats: Statistics from heatmap generation
            nuclei_stats: Statistics from nuclei segmentation
            patch_probabilities: List of cancer probabilities per patch

        Returns:
            GradingFeatures object
        """
        # Cancer probability features
        mean_prob = heatmap_stats.get('mean_probability', 0.5)
        max_prob = heatmap_stats.get('max_probability', 0.5)
        high_risk = heatmap_stats.get('high_risk_fraction', 0.0)

        # Nuclear features
        area_variation = nuclei_stats.get('area_variation', 0.3)
        mean_eccentricity = nuclei_stats.get('mean_eccentricity', 0.5)
        mean_solidity = nuclei_stats.get('mean_solidity', 0.9)
        nucleus_count = nuclei_stats.get('count', 100)
        mean_area = nuclei_stats.get('mean_area', 200)

        # Compute derived scores
        pleomorphism_score = self._compute_pleomorphism(
            area_variation, mean_area, nuclei_stats.get('std_area', 50)
        )

        atypia_score = self._compute_atypia(
            mean_eccentricity, mean_solidity
        )

        mitotic_estimate = self._estimate_mitotic_activity(
            nucleus_count, high_risk, mean_prob
        )

        texture_heterogeneity = self._compute_texture_heterogeneity(
            patch_probabilities
        )

        # Malignant cell density (cells per unit area estimate)
        malignant_density = (nucleus_count * mean_prob) / max(mean_area, 1)

        return GradingFeatures(
            mean_cancer_probability=mean_prob,
            max_cancer_probability=max_prob,
            cancer_area_fraction=high_risk,
            nuclear_pleomorphism_score=pleomorphism_score,
            nuclear_atypia_score=atypia_score,
            mitotic_activity_estimate=mitotic_estimate,
            texture_heterogeneity=texture_heterogeneity,
            malignant_cell_density=malignant_density
        )

    def _compute_pleomorphism(
        self,
        area_variation: float,
        mean_area: float,
        std_area: float
    ) -> float:
        """
        Compute nuclear pleomorphism score (0-1).
        Higher values indicate more variation in nuclear size.
        """
        # Normalised coefficient of variation
        cv = std_area / (mean_area + 1e-8)

        # Map to 0-1 range with sigmoid-like transformation
        score = 1 / (1 + np.exp(-5 * (cv - 0.4)))

        return float(np.clip(score, 0, 1))

    def _compute_atypia(
        self,
        eccentricity: float,
        solidity: float
    ) -> float:
        """
        Compute nuclear atypia score (0-1).
        Based on shape irregularity indicators.
        """
        # Higher eccentricity = more elongated (abnormal)
        # Lower solidity = more irregular boundaries (abnormal)

        eccentricity_score = eccentricity  # Already 0-1
        irregularity_score = 1 - solidity  # Invert solidity

        # Combined score
        score = 0.5 * eccentricity_score + 0.5 * irregularity_score

        return float(np.clip(score, 0, 1))

    def _estimate_mitotic_activity(
        self,
        nucleus_count: int,
        high_risk_fraction: float,
        mean_cancer_prob: float
    ) -> float:
        """
        Estimate mitotic activity score (0-1).
        Based on nuclear density and cancer probability.
        """
        # Higher nuclear density in high-risk regions suggests mitotic activity
        density_factor = min(nucleus_count / 500, 1.0)
        risk_factor = high_risk_fraction * mean_cancer_prob

        score = 0.6 * density_factor + 0.4 * risk_factor

        return float(np.clip(score, 0, 1))

    def _compute_texture_heterogeneity(
        self,
        patch_probabilities: List[float]
    ) -> float:
        """
        Compute texture heterogeneity using probability distribution entropy.
        """
        if not patch_probabilities:
            return 0.5

        probs = np.array(patch_probabilities)

        # Bin probabilities
        hist, _ = np.histogram(probs, bins=10, range=(0, 1), density=True)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()

        # Compute normalised entropy
        max_entropy = np.log(10)  # Maximum for 10 bins
        het = entropy(hist) / max_entropy

        return float(het)

    def estimate_grade(self, features: GradingFeatures) -> GradeEstimate:
        """
        Estimate tumour grade from extracted features.

        Args:
            features: GradingFeatures object

        Returns:
            GradeEstimate with probability distribution
        """
        # Compute raw scores for each grade
        scores = {}

        for grade, weights in self.grade_weights.items():
            score = (
                weights['cancer_prob'] * features.mean_cancer_probability +
                weights['pleomorphism'] * features.nuclear_pleomorphism_score +
                weights['atypia'] * features.nuclear_atypia_score +
                weights['mitotic'] * features.mitotic_activity_estimate +
                weights['heterogeneity'] * features.texture_heterogeneity +
                weights['bias']
            )
            scores[grade] = score

        # Convert to probabilities
        if self.use_softmax:
            exp_scores = {k: np.exp(v) for k, v in scores.items()}
            total = sum(exp_scores.values())
            probs = {k: v / total for k, v in exp_scores.items()}
        else:
            # Simple normalisation
            min_score = min(scores.values())
            shifted = {k: v - min_score + 0.1 for k, v in scores.items()}
            total = sum(shifted.values())
            probs = {k: v / total for k, v in shifted.items()}

        # Determine primary grade
        primary = max(probs, key=probs.get)
        primary_grade = int(primary.split('_')[1])

        # Compute confidence
        sorted_probs = sorted(probs.values(), reverse=True)
        confidence = sorted_probs[0] - sorted_probs[1]

        return GradeEstimate(
            grade_1=round(probs['grade_1'], 3),
            grade_2=round(probs['grade_2'], 3),
            grade_3=round(probs['grade_3'], 3),
            confidence=round(confidence, 3),
            primary_grade=primary_grade,
            features_used=[
                'mean_cancer_probability',
                'nuclear_pleomorphism_score',
                'nuclear_atypia_score',
                'mitotic_activity_estimate',
                'texture_heterogeneity'
            ]
        )


def estimate_grade(
    heatmap_stats: Dict,
    nuclei_stats: Dict,
    patch_probabilities: List[float]
) -> Tuple[GradeEstimate, GradingFeatures]:
    """
    Convenience function to estimate tumour grade.

    Args:
        heatmap_stats: Statistics from heatmap generation
        nuclei_stats: Statistics from nuclei segmentation
        patch_probabilities: List of cancer probabilities

    Returns:
        Tuple of (GradeEstimate, GradingFeatures)
    """
    grader = TumourGrader()
    features = grader.extract_grading_features(
        heatmap_stats, nuclei_stats, patch_probabilities
    )
    estimate = grader.estimate_grade(features)

    return estimate, features


def format_grade_report(estimate: GradeEstimate) -> str:
    """
    Format grade estimate as a readable report.

    Args:
        estimate: GradeEstimate object

    Returns:
        Formatted string report
    """
    report = [
        "TUMOUR GRADE ESTIMATION",
        "=" * 40,
        "",
        f"Primary Grade: Grade {estimate.primary_grade}",
        f"Confidence: {estimate.confidence:.1%}",
        "",
        "Grade Probabilities:",
        f"  Grade 1 (Well-differentiated):     {estimate.grade_1:.1%}",
        f"  Grade 2 (Moderately-differentiated): {estimate.grade_2:.1%}",
        f"  Grade 3 (Poorly-differentiated):   {estimate.grade_3:.1%}",
        "",
        "Note: This estimate is for research demonstration only.",
        "      Not intended for clinical use."
    ]

    return "\n".join(report)
