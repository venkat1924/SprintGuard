"""
Labeling Functions for Weak Supervision
Based on research from Augmenting_NeoDataset.txt

Each LF encodes domain knowledge from software engineering research
to identify risk indicators in user stories.
"""
from snorkel.labeling import labeling_function
import re

# Label constants (Snorkel convention)
RISK = 1      # High Risk
SAFE = 0      # Low Risk
ABSTAIN = -1  # No opinion


# ============================================================================
# Helper Functions
# ============================================================================

def contains_keywords(text, keywords):
    """Case-insensitive keyword matching"""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def count_list_items(text):
    """Count bullet points or numbered lists"""
    if not isinstance(text, str):
        return 0
    return len(re.findall(r'[-*•]\s|\d+\.\s', text))


# ============================================================================
# Lexical LFs (ISO 29148 Ambiguity Indicators)
# Source: Augmenting_NeoDataset.txt Section 3.1, Table 1
# ============================================================================

@labeling_function()
def lf_ambiguity_vague(row):
    """
    Detects subjective quality adjectives (ISO 29148)
    Source: Augmenting_NeoDataset.txt Section 3.1, Table 1
    Rationale: Unverifiable terms lead to scope disputes
    """
    keywords = [
        'user-friendly', 'easy', 'fast', 'robust', 'flexible',
        'efficient', 'seamless', 'intuitive', 'clean'
    ]
    if contains_keywords(row.full_text, keywords):
        return RISK
    return ABSTAIN


@labeling_function()
def lf_ambiguity_loophole(row):
    """
    Detects unbounded scope indicators
    Source: Augmenting_NeoDataset.txt Section 5.1
    Rationale: "Etc" hides undefined requirements
    """
    keywords = [
        'etc', 'et cetera', 'and so on', 
        'including but not limited to', '...'
    ]
    if contains_keywords(row.full_text, keywords):
        return RISK
    return ABSTAIN


@labeling_function()
def lf_quantification_uncertainty(row):
    """
    Detects vague quantifiers
    Source: Augmenting_NeoDataset.txt Section 3.1, Table 1
    Rationale: "Multiple formats" can mean 2 or 20
    """
    keywords = [
        'some', 'several', 'many', 'a few', 'various', 
        'multiple', 'a lot', 'approximately', 'minimal', 'significant'
    ]
    if contains_keywords(row.full_text, keywords):
        return RISK
    return ABSTAIN


@labeling_function()
def lf_weak_action_verbs(row):
    """
    Detects weak action verbs (state vs transformation)
    Source: Augmenting_NeoDataset.txt Table 1
    Rationale: "Handle errors" vs "Log errors to Sentry"
    """
    keywords = [
        'handle', 'support', 'manage', 'facilitate', 
        'enable', 'coordinate', 'improve'
    ]
    if contains_keywords(row.full_text, keywords):
        return RISK
    return ABSTAIN


@labeling_function()
def lf_temporal_ideal(row):
    """
    Detects assumptions about ideal states
    Source: Augmenting_NeoDataset.txt Table 1
    Rationale: Ignores edge cases and error handling
    """
    keywords = [
        'ideally', 'normally', 'usually', 'instantly', 
        'immediately', 'typically'
    ]
    if contains_keywords(row.full_text, keywords):
        return RISK
    return ABSTAIN


# ============================================================================
# Metadata LFs (Story Point Analysis)
# Source: Augmenting_NeoDataset.txt Section 3.2, Section 5.2
# ============================================================================

@labeling_function()
def lf_high_complexity(row):
    """
    Large story points indicate high variance
    Source: Augmenting_NeoDataset.txt Section 5.2
    Threshold: >= 8 points (Cone of Uncertainty principle)
    Rationale: Variance increases exponentially with estimate size
    """
    if row.story_points >= 8:
        return RISK
    return ABSTAIN


@labeling_function()
def lf_low_complexity_safe(row):
    """
    Small stories have low absolute variance
    Source: Augmenting_NeoDataset.txt Section 5.2
    Threshold: <= 2 points
    Rationale: Even if doubled, low absolute impact
    """
    if row.story_points <= 2:
        return SAFE
    return ABSTAIN


@labeling_function()
def lf_fibonacci_anomaly(row):
    """
    Non-Fibonacci points indicate process immaturity
    Source: Augmenting_NeoDataset.txt Section 5.2
    Rationale: Teams treating points as hours (anti-pattern)
    """
    fibonacci_sequence = [1, 2, 3, 5, 8, 13, 20, 40]
    if row.story_points not in fibonacci_sequence:
        return RISK
    return ABSTAIN


# ============================================================================
# Structural LFs (Syntactic Completeness)
# Source: Augmenting_NeoDataset.txt Section 3.3, Section 5.3
# ============================================================================

@labeling_function()
def lf_missing_acceptance_criteria(row):
    """
    Stories without acceptance criteria or list structure
    Source: Augmenting_NeoDataset.txt Section 5.3
    Rationale: No "Definition of Done" = hypothesis, not requirement
    """
    has_ac = 'acceptance criteria' in str(row.description).lower()
    has_lists = row.list_item_count >= 3
    
    if not has_ac and not has_lists:
        return RISK
    return ABSTAIN


@labeling_function()
def lf_dependency_link(row):
    """
    Dependencies introduce blocking risks
    Source: Augmenting_NeoDataset.txt Section 5.3
    Rationale: Exogenous risk independent of story complexity
    """
    pattern = r'(blocked by|depends on|relates to|dependency|blocking)\s*#?\d*'
    if re.search(pattern, row.full_text, re.IGNORECASE):
        return RISK
    return ABSTAIN


@labeling_function()
def lf_very_short_description(row):
    """
    Extremely short descriptions lack detail
    Source: Augmenting_NeoDataset.txt Section 3.2 (Cluster 0 phenomenon)
    Threshold: <15 words
    Rationale: Insufficient analysis performed
    """
    if row.word_count < 15:
        return RISK
    return ABSTAIN


@labeling_function()
def lf_very_long_description(row):
    """
    Overly long descriptions indicate confusion
    Source: Augmenting_NeoDataset.txt Section 3.2 (Cluster 0 phenomenon)
    Threshold: >200 words
    Rationale: Unclear boundaries or merged requirements
    """
    if row.word_count > 200:
        return RISK
    return ABSTAIN


# ============================================================================
# Domain-Specific LFs (Software Engineering Patterns)
# Source: General software engineering research
# ============================================================================

@labeling_function()
def lf_integration_keywords(row):
    """
    Integration tasks have high uncertainty
    Rationale: External dependencies, API changes, auth flows
    """
    keywords = [
        'api integration', 'third-party', 'external api',
        'webhook', 'oauth', 'sso', 'authentication'
    ]
    if contains_keywords(row.full_text, keywords):
        return RISK
    return ABSTAIN


@labeling_function()
def lf_legacy_refactor(row):
    """
    Legacy/refactor work is unpredictable
    Rationale: Hidden complexity in old code, no tests
    """
    keywords = [
        'legacy', 'refactor', 'technical debt', 'rewrite',
        'deprecated', 'monolith'
    ]
    if contains_keywords(row.full_text, keywords):
        return RISK
    return ABSTAIN


@labeling_function()
def lf_security_keywords(row):
    """
    Security features have hidden complexity
    Rationale: Edge cases, attack vectors, compliance
    """
    keywords = [
        'security', 'vulnerability', 'encryption',
        'penetration test', 'audit'
    ]
    if contains_keywords(row.full_text, keywords):
        return RISK
    return ABSTAIN


@labeling_function()
def lf_performance_keywords(row):
    """
    Performance optimization is unpredictable
    Rationale: Requires profiling, testing, iterative tuning
    """
    keywords = [
        'performance', 'optimization', 'scalability',
        'bottleneck', 'latency', 'memory leak'
    ]
    if contains_keywords(row.full_text, keywords):
        return RISK
    return ABSTAIN


@labeling_function()
def lf_bug_fix(row):
    """
    Bug fixes are typically well-scoped
    Rationale: Clear problem statement, testable outcome
    """
    keywords = ['fix', 'bug', 'hotfix', 'bugfix', 'broken', 'error']
    title_lower = str(row.title).lower()
    if any(kw in title_lower for kw in keywords):
        return SAFE
    return ABSTAIN


@labeling_function()
def lf_documentation_story(row):
    """
    Documentation stories are low risk
    Rationale: No code changes, clear deliverable
    """
    keywords = [
        'documentation', 'readme', 'docs', 'comments',
        'docstring', 'guide', 'tutorial'
    ]
    if contains_keywords(row.full_text, keywords):
        return SAFE
    return ABSTAIN


# ============================================================================
# Collect All LFs
# ============================================================================

ALL_LABELING_FUNCTIONS = [
    # Lexical (ISO 29148)
    lf_ambiguity_vague,
    lf_ambiguity_loophole,
    lf_quantification_uncertainty,
    lf_weak_action_verbs,
    lf_temporal_ideal,
    
    # Metadata
    lf_high_complexity,
    lf_low_complexity_safe,
    lf_fibonacci_anomaly,
    
    # Structural
    lf_missing_acceptance_criteria,
    lf_dependency_link,
    lf_very_short_description,
    lf_very_long_description,
    
    # Domain-specific
    lf_integration_keywords,
    lf_legacy_refactor,
    lf_security_keywords,
    lf_performance_keywords,
    lf_bug_fix,
    lf_documentation_story,
]

