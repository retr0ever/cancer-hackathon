"""
Explanation Generation Module for PathoLens

Generates human-readable explanations of model findings using
OpenAI's API to interpret histopathological features.

IMPORTANT: This module requires an OpenAI API key to function.
Set your key in a .env file or environment variable.
"""

import os
from typing import Dict, Optional, List
from dataclasses import dataclass

# TODO: Install python-dotenv and uncomment when ready
# from dotenv import load_dotenv
# load_dotenv()


@dataclass
class AnalysisFindings:
    """Structured findings from the analysis pipeline."""
    cancer_probability: float
    malignant_cell_count: int
    malignant_cell_count_uncertainty: float
    grade_estimate: Dict[str, float]
    primary_grade: int
    high_risk_regions: int
    nuclear_pleomorphism: str  # 'low', 'moderate', 'high'
    nuclear_atypia: str  # 'mild', 'moderate', 'severe'
    tissue_heterogeneity: str  # 'homogeneous', 'heterogeneous', 'highly heterogeneous'
    hotspot_locations: List[str]  # Descriptive locations


def get_openai_client():
    """
    Initialise and return OpenAI client.

    TODO: Set your OpenAI API key in one of these ways:
    1. Create a .env file with: OPENAI_API_KEY=your-key-here
    2. Set environment variable: export OPENAI_API_KEY=your-key-here
    """
    try:
        from openai import OpenAI

        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            return None

        return OpenAI(api_key=api_key)
    except ImportError:
        print("OpenAI package not installed. Run: pip install openai")
        return None


def build_analysis_prompt(findings: AnalysisFindings) -> str:
    """
    Build a prompt for the OpenAI API based on analysis findings.

    Args:
        findings: Structured findings from the analysis

    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a computational pathology assistant helping to explain
histopathology analysis results to researchers. Based on the following automated
analysis findings, provide a clear, professional explanation that a histopathologist
would find useful. Use appropriate medical terminology but ensure clarity.

ANALYSIS FINDINGS:
- Overall Cancer Probability: {findings.cancer_probability:.1%}
- Estimated Malignant Cell Count: {findings.malignant_cell_count} (±{findings.malignant_cell_count_uncertainty:.0%})
- Grade Distribution: Grade 1: {findings.grade_estimate.get('grade_1', 0):.1%}, Grade 2: {findings.grade_estimate.get('grade_2', 0):.1%}, Grade 3: {findings.grade_estimate.get('grade_3', 0):.1%}
- Primary Grade Estimate: Grade {findings.primary_grade}
- High-Risk Regions Detected: {findings.high_risk_regions}
- Nuclear Pleomorphism: {findings.nuclear_pleomorphism}
- Nuclear Atypia: {findings.nuclear_atypia}
- Tissue Heterogeneity: {findings.tissue_heterogeneity}
- Notable Hotspot Locations: {', '.join(findings.hotspot_locations) if findings.hotspot_locations else 'None identified'}

Please provide:
1. A brief summary of what the analysis detected (2-3 sentences)
2. Interpretation of the key histopathological patterns observed
3. What the grade estimate suggests about tissue differentiation
4. Any notable patterns that warrant attention
5. Standard disclaimer about this being a computational analysis for research

Keep the response professional, concise, and suitable for a research context.
Do not provide clinical recommendations or diagnostic conclusions."""

    return prompt


def generate_explanation(
    model_features: Dict,
    use_api: bool = True
) -> str:
    """
    Generate an explanation of the model's findings.

    Args:
        model_features: Dictionary containing analysis results with keys:
            - cancer_probability: float
            - malignant_cell_count: int
            - malignant_cell_uncertainty: float
            - grade_1, grade_2, grade_3: float (probabilities)
            - primary_grade: int
            - high_risk_regions: int
            - nuclear_pleomorphism_score: float
            - nuclear_atypia_score: float
            - texture_heterogeneity: float
            - hotspot_locations: list

    Returns:
        Human-readable explanation string
    """
    # Convert features to structured findings
    findings = _parse_features_to_findings(model_features)

    if use_api:
        explanation = _generate_with_api(findings)
        if explanation:
            return explanation

    # Fallback to template-based explanation
    return _generate_template_explanation(findings)


def _parse_features_to_findings(features: Dict) -> AnalysisFindings:
    """Convert raw feature dictionary to structured findings."""

    # Map numerical scores to categorical descriptions
    pleomorphism_score = features.get('nuclear_pleomorphism_score', 0.5)
    if pleomorphism_score < 0.33:
        pleomorphism = 'low'
    elif pleomorphism_score < 0.66:
        pleomorphism = 'moderate'
    else:
        pleomorphism = 'high'

    atypia_score = features.get('nuclear_atypia_score', 0.5)
    if atypia_score < 0.33:
        atypia = 'mild'
    elif atypia_score < 0.66:
        atypia = 'moderate'
    else:
        atypia = 'severe'

    heterogeneity = features.get('texture_heterogeneity', 0.5)
    if heterogeneity < 0.33:
        het_desc = 'homogeneous'
    elif heterogeneity < 0.66:
        het_desc = 'heterogeneous'
    else:
        het_desc = 'highly heterogeneous'

    return AnalysisFindings(
        cancer_probability=features.get('cancer_probability', 0.0),
        malignant_cell_count=features.get('malignant_cell_count', 0),
        malignant_cell_count_uncertainty=features.get('malignant_cell_uncertainty', 0.15),
        grade_estimate={
            'grade_1': features.get('grade_1', 0.33),
            'grade_2': features.get('grade_2', 0.34),
            'grade_3': features.get('grade_3', 0.33)
        },
        primary_grade=features.get('primary_grade', 2),
        high_risk_regions=features.get('high_risk_regions', 0),
        nuclear_pleomorphism=pleomorphism,
        nuclear_atypia=atypia,
        tissue_heterogeneity=het_desc,
        hotspot_locations=features.get('hotspot_locations', [])
    )


def _generate_with_api(findings: AnalysisFindings) -> Optional[str]:
    """
    Generate explanation using OpenAI API.

    TODO: Ensure your OpenAI API key is set before using this function.
    """
    client = get_openai_client()

    if client is None:
        return None

    prompt = build_analysis_prompt(findings)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective model for explanations
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert computational pathology assistant that explains histopathology analysis results clearly and professionally."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=800,
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None


def _generate_template_explanation(findings: AnalysisFindings) -> str:
    """
    Generate a template-based explanation when API is unavailable.

    This provides a structured but less dynamic explanation.
    """
    # Determine severity descriptors
    if findings.cancer_probability < 0.3:
        severity = "low"
        concern = "minimal concerning features"
    elif findings.cancer_probability < 0.6:
        severity = "moderate"
        concern = "some features warranting attention"
    else:
        severity = "elevated"
        concern = "significant features of interest"

    # Build explanation
    explanation_parts = [
        "COMPUTATIONAL ANALYSIS SUMMARY",
        "=" * 50,
        "",
        f"The automated analysis of this histopathology section reveals {concern} "
        f"with an overall malignancy probability of {findings.cancer_probability:.1%}.",
        "",
        "KEY FINDINGS:",
        "",
        f"1. CELLULAR ANALYSIS",
        f"   - Estimated malignant cell count: {findings.malignant_cell_count} "
        f"(±{findings.malignant_cell_count_uncertainty:.0%})",
        f"   - Nuclear pleomorphism: {findings.nuclear_pleomorphism}",
        f"   - Nuclear atypia: {findings.nuclear_atypia}",
        "",
        f"2. GRADE ESTIMATION",
        f"   The analysis suggests a Grade {findings.primary_grade} pattern as most likely:",
        f"   - Grade 1 (well-differentiated): {findings.grade_estimate['grade_1']:.1%}",
        f"   - Grade 2 (moderately-differentiated): {findings.grade_estimate['grade_2']:.1%}",
        f"   - Grade 3 (poorly-differentiated): {findings.grade_estimate['grade_3']:.1%}",
        "",
        f"3. SPATIAL PATTERNS",
        f"   - Tissue heterogeneity: {findings.tissue_heterogeneity}",
        f"   - High-risk regions identified: {findings.high_risk_regions}",
    ]

    if findings.hotspot_locations:
        explanation_parts.append(f"   - Notable hotspots: {', '.join(findings.hotspot_locations)}")

    explanation_parts.extend([
        "",
        "INTERPRETATION:",
        "",
    ])

    # Grade-specific interpretation
    if findings.primary_grade == 1:
        explanation_parts.append(
            "The predominant Grade 1 pattern suggests well-differentiated tissue "
            "with relatively preserved cellular architecture. Nuclear features show "
            "limited pleomorphism and atypia."
        )
    elif findings.primary_grade == 2:
        explanation_parts.append(
            "The predominant Grade 2 pattern indicates moderately-differentiated tissue "
            "with intermediate nuclear features. There is evidence of some architectural "
            "disruption and nuclear irregularity."
        )
    else:
        explanation_parts.append(
            "The predominant Grade 3 pattern suggests poorly-differentiated tissue "
            "with significant nuclear pleomorphism and atypia. The cellular architecture "
            "shows substantial disorganisation."
        )

    explanation_parts.extend([
        "",
        "-" * 50,
        "DISCLAIMER: This analysis is generated by a computational model for "
        "research purposes only. It is not intended for clinical diagnosis or "
        "treatment decisions. All findings should be reviewed and validated by "
        "a qualified pathologist.",
        "-" * 50
    ])

    return "\n".join(explanation_parts)


def format_features_for_display(features: Dict) -> str:
    """
    Format feature dictionary for display in the UI.

    Args:
        features: Raw feature dictionary

    Returns:
        Formatted string for display
    """
    lines = [
        "EXTRACTED FEATURES",
        "-" * 30,
        f"Cancer Probability: {features.get('cancer_probability', 0):.2%}",
        f"Malignant Cells: {features.get('malignant_cell_count', 0)}",
        f"High-Risk Regions: {features.get('high_risk_regions', 0)}",
        f"Nuclear Pleomorphism: {features.get('nuclear_pleomorphism_score', 0):.2f}",
        f"Nuclear Atypia: {features.get('nuclear_atypia_score', 0):.2f}",
        f"Texture Heterogeneity: {features.get('texture_heterogeneity', 0):.2f}",
    ]

    return "\n".join(lines)
