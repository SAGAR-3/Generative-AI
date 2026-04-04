"""
src/guardrails/compliance_checker.py
======================================
Banking regulatory compliance checker for RAG outputs.

Regulations covered:
- ECOA (Equal Credit Opportunity Act / Regulation B): Anti-discrimination
- TILA (Truth in Lending Act / Regulation Z): Rate disclosure
- RESPA (Real Estate Settlement Procedures Act / Regulation X): Settlement
- FCRA (Fair Credit Reporting Act): Credit information usage
- UDAAP (Unfair, Deceptive, or Abusive Acts or Practices)

This module checks BOTH inputs (queries) and outputs (LLM responses)
for compliance violations before they reach customers.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)


# ─── Compliance Result ────────────────────────────────────────────────────────

class ViolationType(str, Enum):
    ECOA_DISCRIMINATION = "ECOA_DISCRIMINATION"
    TILA_RATE_GUARANTEE = "TILA_RATE_GUARANTEE"
    RESPA_KICKBACK = "RESPA_KICKBACK"
    FCRA_MISUSE = "FCRA_MISUSE"
    UDAAP_DECEPTIVE = "UDAAP_DECEPTIVE"
    UDAAP_UNFAIR = "UDAAP_UNFAIR"
    RATE_GUARANTEE = "RATE_GUARANTEE"
    APPROVAL_GUARANTEE = "APPROVAL_GUARANTEE"
    UNAUTHORIZED_ADVICE = "UNAUTHORIZED_ADVICE"
    DATA_PRIVACY = "DATA_PRIVACY"


class Severity(str, Enum):
    CRITICAL = "critical"   # Block response, alert compliance team
    HIGH = "high"           # Block response, log for review
    MEDIUM = "medium"       # Allow with modification/disclaimer
    LOW = "low"             # Log for review, allow


@dataclass
class ComplianceViolation:
    violation_type: ViolationType
    severity: Severity
    description: str
    matched_text: str
    regulation: str
    recommendation: str


@dataclass
class ComplianceCheckResult:
    text: str
    violations: List[ComplianceViolation]
    is_compliant: bool
    should_block: bool
    modified_text: Optional[str] = None
    disclaimer_added: bool = False

    @property
    def critical_violations(self) -> List[ComplianceViolation]:
        return [v for v in self.violations if v.severity == Severity.CRITICAL]

    @property
    def high_violations(self) -> List[ComplianceViolation]:
        return [v for v in self.violations if v.severity == Severity.HIGH]


# ─── Compliance Patterns ──────────────────────────────────────────────────────

# ECOA/Regulation B: Discriminatory language based on protected classes
ECOA_PROTECTED_CLASSES = [
    "race", "color", "religion", "national origin", "sex", "gender",
    "marital status", "age", "family status", "receipt of public assistance",
    "disability", "ethnicity", "pregnancy"
]

ECOA_DISCRIMINATORY_PATTERNS = [
    re.compile(
        r'\b(?:we|bank|lender)\s+(?:don\'t|do not|won\'t|will not|cannot|can\'t)\s+'
        r'(?:lend|loan|approve|finance|offer)\s+to\s+(?:' +
        '|'.join(ECOA_PROTECTED_CLASSES) + r')',
        re.IGNORECASE
    ),
    re.compile(
        r'\b(?:' + '|'.join(ECOA_PROTECTED_CLASSES) + r')\s+'
        r'(?:people|individuals|borrowers|customers|applicants)\s+'
        r'(?:are|tend to be|are more likely|have higher)',
        re.IGNORECASE
    ),
    re.compile(
        r'\bwe\s+(?:prefer|prefer\s+not|don\'t\s+like)\s+'
        r'(?:' + '|'.join(ECOA_PROTECTED_CLASSES) + r')',
        re.IGNORECASE
    ),
]

# TILA/Regulation Z: Rate guarantees without proper disclosures
RATE_GUARANTEE_PATTERNS = [
    re.compile(
        r'\b(?:your|the|guaranteed?|locked?)\s+(?:interest\s+)?rate\s+(?:is|will\s+be|shall\s+be)\s+'
        r'(?:definitely|absolutely|exactly)?\s*[\d\.]+\s*%',
        re.IGNORECASE
    ),
    re.compile(
        r'\bi\s+(?:can|will|am\s+able\s+to)\s+(?:guarantee|promise|assure)\s+you\s+'
        r'(?:a|an|the)?\s*(?:rate|apr|interest)',
        re.IGNORECASE
    ),
    re.compile(r'\bguaranteed\s+(?:low\s+)?rate', re.IGNORECASE),
]

# Approval guarantees (misleading)
APPROVAL_GUARANTEE_PATTERNS = [
    re.compile(
        r'\b(?:you\s+(?:will|are)\s+(?:definitely|certainly|absolutely)?\s*(?:be\s+)?approved)',
        re.IGNORECASE
    ),
    re.compile(r'\bguaranteed\s+(?:loan\s+)?approval', re.IGNORECASE),
    re.compile(r'\b100%\s+(?:approval|approv(?:ed|al))', re.IGNORECASE),
    re.compile(r'\bno\s+(?:one\s+is\s+)?(?:ever\s+)?denied', re.IGNORECASE),
]

# UDAAP: Deceptive claims
UDAAP_DECEPTIVE_PATTERNS = [
    re.compile(r'\bno\s+(?:hidden\s+)?fees?\s+(?:ever|at\s+all|whatsoever)', re.IGNORECASE),
    re.compile(r'\bcompletely\s+free\s+(?:loan|mortgage|refinance)', re.IGNORECASE),
    re.compile(r'\bno\s+credit\s+check\s+(?:required|needed|necessary)', re.IGNORECASE),
    re.compile(r'\binstant\s+(?:approval|decision)\s+guaranteed', re.IGNORECASE),
    re.compile(r'\blowest\s+(?:rates?\s+)?(?:in\s+(?:the\s+)?)?(?:town|state|country|nation)', re.IGNORECASE),
]

# Legal/financial advice overstepping
UNAUTHORIZED_ADVICE_PATTERNS = [
    re.compile(
        r'\byou\s+(?:should|must|need\s+to|have\s+to)\s+'
        r'(?:file\s+for\s+bankruptcy|declare\s+bankruptcy)',
        re.IGNORECASE
    ),
    re.compile(
        r'\bfor\s+tax\s+purposes?,\s+you\s+(?:should|must)',
        re.IGNORECASE
    ),
    re.compile(
        r'\byour\s+(?:legal|tax|financial)\s+(?:strategy|plan|approach)\s+should\s+be',
        re.IGNORECASE
    ),
]

# Required disclaimers for certain topics
DISCLAIMER_TRIGGERS = {
    re.compile(r'\b(?:interest\s+)?rate[s]?\b', re.IGNORECASE):
        "\n\n*Rates are subject to change and are not a commitment to lend. "
        "Contact your loan officer for a personalized rate quote.",

    re.compile(r'\b(?:loan\s+)?approval\b', re.IGNORECASE):
        "\n\n*Loan approval is subject to full application review, "
        "underwriting approval, and credit qualification.",

    re.compile(r'\bpmi|private\s+mortgage\s+insurance\b', re.IGNORECASE):
        "\n\n*PMI requirements are based on individual loan terms and LTV ratios.",
}


# ─── Compliance Checker ───────────────────────────────────────────────────────

class ComplianceChecker:
    """
    Banking regulatory compliance checker.
    
    Checks both:
    - INPUT (user queries): Detect inappropriate requests
    - OUTPUT (LLM responses): Ensure responses are compliant

    In production, augment with:
    - ML classifier fine-tuned on CFPB enforcement actions
    - Legal team-approved response templates
    - Real-time regulatory update feeds
    """

    def __init__(
        self,
        add_disclaimers: bool = True,
        block_on_severity: Severity = Severity.HIGH,
    ):
        self.add_disclaimers = add_disclaimers
        self.block_on_severity = block_on_severity
        self._severity_order = {
            Severity.LOW: 0,
            Severity.MEDIUM: 1,
            Severity.HIGH: 2,
            Severity.CRITICAL: 3,
        }

    def check(self, text: str, check_type: str = "output") -> ComplianceCheckResult:
        """
        Check text for compliance violations.

        Args:
            text: Text to check (query or LLM response)
            check_type: "input" (query) or "output" (response)

        Returns:
            ComplianceCheckResult
        """
        violations = []

        # ECOA checks
        for pattern in ECOA_DISCRIMINATORY_PATTERNS:
            if pattern.search(text):
                violations.append(ComplianceViolation(
                    violation_type=ViolationType.ECOA_DISCRIMINATION,
                    severity=Severity.CRITICAL,
                    description="Response contains potentially discriminatory language",
                    matched_text=pattern.search(text).group(0),
                    regulation="ECOA (12 CFR Part 1002 - Regulation B)",
                    recommendation="Remove discriminatory language. Never reference protected classes in lending decisions.",
                ))

        # Rate guarantee checks
        for pattern in RATE_GUARANTEE_PATTERNS:
            match = pattern.search(text)
            if match:
                violations.append(ComplianceViolation(
                    violation_type=ViolationType.RATE_GUARANTEE,
                    severity=Severity.HIGH,
                    description="Response appears to guarantee a specific interest rate without proper TILA disclosures",
                    matched_text=match.group(0),
                    regulation="TILA (15 U.S.C. 1638 - Regulation Z)",
                    recommendation="Add disclaimer: 'Rates are subject to change and not a commitment to lend.' "
                                   "Remove absolute rate guarantees.",
                ))

        # Approval guarantee checks
        for pattern in APPROVAL_GUARANTEE_PATTERNS:
            match = pattern.search(text)
            if match:
                violations.append(ComplianceViolation(
                    violation_type=ViolationType.APPROVAL_GUARANTEE,
                    severity=Severity.HIGH,
                    description="Response appears to guarantee loan approval",
                    matched_text=match.group(0),
                    regulation="UDAAP (12 U.S.C. 5531)",
                    recommendation="Replace with: 'Subject to credit approval and underwriting review.'",
                ))

        # UDAAP deceptive practices
        for pattern in UDAAP_DECEPTIVE_PATTERNS:
            match = pattern.search(text)
            if match:
                violations.append(ComplianceViolation(
                    violation_type=ViolationType.UDAAP_DECEPTIVE,
                    severity=Severity.HIGH,
                    description="Response may contain deceptive or misleading claims",
                    matched_text=match.group(0),
                    regulation="UDAAP (12 U.S.C. 5531, CFPB Supervision Manual)",
                    recommendation="Remove absolute claims. All marketing must be accurate and not misleading.",
                ))

        # Unauthorized advice
        for pattern in UNAUTHORIZED_ADVICE_PATTERNS:
            match = pattern.search(text)
            if match:
                violations.append(ComplianceViolation(
                    violation_type=ViolationType.UNAUTHORIZED_ADVICE,
                    severity=Severity.MEDIUM,
                    description="Response may contain unauthorized legal or financial advice",
                    matched_text=match.group(0),
                    regulation="Various (unauthorized practice of law, financial advice regulations)",
                    recommendation="Add: 'Please consult a qualified attorney or financial advisor for personalized guidance.'",
                ))

        # Determine if should block
        threshold = self._severity_order[self.block_on_severity]
        should_block = any(
            self._severity_order[v.severity] >= threshold
            for v in violations
        )

        # Add required disclaimers
        modified_text = text
        disclaimer_added = False
        if self.add_disclaimers and not should_block:
            for pattern, disclaimer in DISCLAIMER_TRIGGERS.items():
                if pattern.search(text) and disclaimer not in text:
                    modified_text += disclaimer
                    disclaimer_added = True
                    break  # Add one disclaimer max per response

        is_compliant = len(violations) == 0

        if violations:
            logger.warning(
                "compliance_violations_found",
                count=len(violations),
                types=[v.violation_type.value for v in violations],
                should_block=should_block,
                check_type=check_type,
            )

        return ComplianceCheckResult(
            text=text,
            violations=violations,
            is_compliant=is_compliant,
            should_block=should_block,
            modified_text=modified_text if disclaimer_added else None,
            disclaimer_added=disclaimer_added,
        )

    def check_query(self, query: str) -> ComplianceCheckResult:
        """Check an incoming user query for compliance issues."""
        return self.check(query, check_type="input")

    def check_response(self, response: str) -> ComplianceCheckResult:
        """Check an outgoing LLM response for compliance issues."""
        return self.check(response, check_type="output")
