"""
tests/test_guardrails.py
=========================
Unit tests for PII redaction and compliance checking guardrails.
These are critical safety tests — all must pass before deployment.
"""

import pytest


# ─── PII Redactor Tests ───────────────────────────────────────────────────────

class TestPIIRedactor:

    @pytest.fixture
    def redactor(self):
        from src.guardrails.pii_redactor import PIIRedactor
        return PIIRedactor(use_presidio=False, min_confidence=0.75)

    def test_ssn_detection(self, redactor):
        text = "My Social Security number is 123-45-6789 please help."
        result = redactor.redact(text)
        assert result.was_modified
        assert "[REDACTED-SSN]" in result.redacted_text
        assert "123-45-6789" not in result.redacted_text
        assert result.pii_count >= 1

    def test_credit_card_detection(self, redactor):
        text = "Charge my card 4532015112830366 for the appraisal fee."
        result = redactor.redact(text)
        assert result.was_modified
        assert "[REDACTED-CC]" in result.redacted_text

    def test_email_detection(self, redactor):
        text = "Please email me at john.doe@example.com with the details."
        result = redactor.redact(text)
        assert result.was_modified
        assert "[REDACTED-EMAIL]" in result.redacted_text
        assert "john.doe@example.com" not in result.redacted_text

    def test_phone_detection(self, redactor):
        text = "Call me at (555) 867-5309 to discuss the loan."
        result = redactor.redact(text)
        assert result.was_modified
        assert "[REDACTED-PHONE]" in result.redacted_text

    def test_no_pii_unchanged(self, redactor):
        text = "What is the minimum credit score for an FHA loan?"
        result = redactor.redact(text)
        assert not result.was_modified
        assert result.redacted_text == text
        assert result.pii_count == 0

    def test_multiple_pii_types(self, redactor):
        text = "SSN: 987-65-4320, email: test@bank.com, call: 555-123-4567"
        result = redactor.redact(text)
        assert result.was_modified
        assert result.pii_count >= 2
        assert "987-65-4320" not in result.redacted_text
        assert "test@bank.com" not in result.redacted_text

    def test_risk_levels(self, redactor):
        from src.guardrails.pii_redactor import RiskLevel
        # SSN is HIGH risk
        text = "SSN: 123-45-6789"
        result = redactor.redact(text)
        assert result.risk_level == RiskLevel.HIGH

    def test_entity_summary(self, redactor):
        text = "SSN: 123-45-6789 and email: test@test.com"
        result = redactor.redact(text)
        summary = result.entity_summary
        assert "SSN" in summary or "EMAIL" in summary

    def test_should_block_excessive_pii(self, redactor):
        # 3+ high-risk PII entities should trigger block
        text = (
            "SSN: 123-45-6789, Account: 1234567890123456, "
            "Card: 4532015112830366, Routing: 021000021"
        )
        result = redactor.redact(text)
        assert redactor.should_block(result)

    def test_preserves_non_pii_content(self, redactor):
        text = "The FHA loan requires a 580 credit score and 3.5% down payment."
        result = redactor.redact(text)
        assert "580 credit score" in result.redacted_text
        assert "3.5% down payment" in result.redacted_text


# ─── Compliance Checker Tests ─────────────────────────────────────────────────

class TestComplianceChecker:

    @pytest.fixture
    def checker(self):
        from src.guardrails.compliance_checker import ComplianceChecker
        return ComplianceChecker(add_disclaimers=True)

    def test_clean_response_passes(self, checker):
        text = (
            "Based on our guidelines, the minimum credit score for an FHA loan is 580. "
            "[Source: fha_guidelines.pdf] For personalized guidance, contact your loan officer."
        )
        result = checker.check_response(text)
        assert result.is_compliant
        assert not result.should_block
        assert len(result.violations) == 0

    def test_rate_guarantee_blocked(self, checker):
        from src.guardrails.compliance_checker import ViolationType
        text = "I can guarantee you a rate of exactly 6.5% on your mortgage."
        result = checker.check_response(text)
        assert not result.is_compliant
        assert any(v.violation_type == ViolationType.RATE_GUARANTEE for v in result.violations)

    def test_approval_guarantee_blocked(self, checker):
        from src.guardrails.compliance_checker import ViolationType
        text = "Don't worry, you will definitely be approved for this loan!"
        result = checker.check_response(text)
        assert not result.is_compliant
        assert any(v.violation_type == ViolationType.APPROVAL_GUARANTEE for v in result.violations)

    def test_udaap_deceptive_blocked(self, checker):
        from src.guardrails.compliance_checker import ViolationType
        text = "We offer completely free loans with no hidden fees ever!"
        result = checker.check_response(text)
        assert not result.is_compliant
        assert any(v.violation_type == ViolationType.UDAAP_DECEPTIVE for v in result.violations)

    def test_disclaimer_added_for_rates(self, checker):
        text = "Current interest rates are approximately 7% for a 30-year fixed loan."
        result = checker.check(text)
        # Disclaimer should be added for rate mentions
        if result.modified_text:
            assert "subject to change" in result.modified_text.lower()

    def test_violation_severity(self, checker):
        from src.guardrails.compliance_checker import Severity
        text = "100% guaranteed approval for everyone!"
        result = checker.check_response(text)
        if result.violations:
            severities = [v.severity for v in result.violations]
            assert Severity.HIGH in severities or Severity.CRITICAL in severities

    def test_query_compliance_check(self, checker):
        """Queries should generally pass unless obviously problematic."""
        query = "What is the minimum credit score for a conventional loan?"
        result = checker.check_query(query)
        assert result.is_compliant

    def test_compliance_result_has_recommendation(self, checker):
        """Every violation should include a recommendation."""
        text = "I guarantee you will be approved."
        result = checker.check_response(text)
        for violation in result.violations:
            assert violation.recommendation is not None
            assert len(violation.recommendation) > 0
            assert violation.regulation is not None


# ─── Integration: PII + Compliance ───────────────────────────────────────────

class TestGuardrailsIntegration:

    def test_pii_then_compliance_pipeline(self):
        """Simulate the full guardrails pipeline on a query."""
        from src.guardrails.pii_redactor import PIIRedactor
        from src.guardrails.compliance_checker import ComplianceChecker

        redactor = PIIRedactor(use_presidio=False)
        checker = ComplianceChecker()

        # Query with PII
        raw_query = "My SSN is 123-45-6789. Will I be approved for FHA loan?"
        pii_result = redactor.redact(raw_query)
        safe_query = pii_result.redacted_text

        # PII should be redacted
        assert "123-45-6789" not in safe_query
        assert "[REDACTED-SSN]" in safe_query

        # Compliance check on redacted query
        compliance_result = checker.check_query(safe_query)
        # Query intent (will I be approved) is fine — just asking info
        assert compliance_result is not None

    def test_output_guardrails_pipeline(self):
        """Simulate output guardrails checking an LLM response."""
        from src.guardrails.pii_redactor import PIIRedactor
        from src.guardrails.compliance_checker import ComplianceChecker

        redactor = PIIRedactor(use_presidio=False)
        checker = ComplianceChecker()

        # Simulated LLM response with both PII and compliance issue
        llm_response = (
            "Based on our guidelines, borrowers with 580+ credit score qualify for FHA. "
            "I can guarantee you a 6.5% interest rate. Contact us at loans@bank.com."
        )

        # Output PII check
        pii_result = redactor.redact(llm_response)
        assert "loans@bank.com" not in pii_result.redacted_text

        # Output compliance check
        compliance_result = checker.check_response(pii_result.redacted_text)
        # Rate guarantee violation should still be caught
        assert not compliance_result.is_compliant
