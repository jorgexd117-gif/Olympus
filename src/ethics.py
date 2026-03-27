"""Ethics framework for the AI agent system.

Provides guardrails that filter agent inputs and outputs to ensure
responsible, safe, and ethical behavior across all sub-agents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EthicsViolationType(str, Enum):
    HARMFUL_CONTENT = "harmful_content"
    PRIVACY_VIOLATION = "privacy_violation"
    DECEPTION = "deception"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    BIAS_DETECTED = "bias_detected"
    UNSAFE_OUTPUT = "unsafe_output"


@dataclass
class EthicsCheckResult:
    is_safe: bool
    violations: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sanitized_content: str = ""
    applied_rules: list[str] = field(default_factory=list)


ETHICS_PRINCIPLES = {
    "beneficence": (
        "Act in the best interest of the user and society. "
        "Prioritize helpful, accurate, and constructive responses."
    ),
    "non_maleficence": (
        "Do not generate content that could cause harm. "
        "Refuse requests for dangerous, illegal, or destructive actions."
    ),
    "autonomy": (
        "Respect the user's right to make informed decisions. "
        "Present options and trade-offs rather than imposing choices."
    ),
    "transparency": (
        "Be honest about limitations and uncertainties. "
        "Clearly indicate when information may be incomplete or speculative."
    ),
    "fairness": (
        "Avoid bias in analysis and recommendations. "
        "Consider diverse perspectives and impacts."
    ),
    "privacy": (
        "Protect sensitive information. "
        "Do not expose credentials, personal data, or internal system details."
    ),
    "accountability": (
        "Provide traceable reasoning. "
        "Each decision should reference the principle that guided it."
    ),
}

_BLOCKED_INPUT_PATTERNS = [
    (r"(?i)\b(hack|exploit|crack)\s+(password|system|server|database)", EthicsViolationType.HARMFUL_CONTENT),
    (r"(?i)\b(steal|exfiltrate|dump)\s+(data|credentials|passwords|tokens)", EthicsViolationType.HARMFUL_CONTENT),
    (r"(?i)\b(inject|injection)\s+(sql|xss|code|script)", EthicsViolationType.HARMFUL_CONTENT),
    (r"(?i)ignore\s+(previous|all|prior)\s+(instructions|rules|constraints)", EthicsViolationType.DECEPTION),
    (r"(?i)pretend\s+you\s+are\s+(?:not\s+)?(?:an?\s+)?(?:ai|bot|assistant)", EthicsViolationType.DECEPTION),
    (r"(?i)bypass\s+(safety|security|ethics|filter|guardrail)", EthicsViolationType.DECEPTION),
]

_BLOCKED_OUTPUT_PATTERNS = [
    (r"(?i)\b(password|api_key|secret_key|private_key)\s*[:=]\s*\S+", EthicsViolationType.PRIVACY_VIOLATION),
    (r"(?i)BEGIN\s+(RSA|DSA|EC|OPENSSH)\s+PRIVATE\s+KEY", EthicsViolationType.PRIVACY_VIOLATION),
    (r"(?i)(ssn|social\s+security)\s*[:=]?\s*\d{3}[-\s]?\d{2}[-\s]?\d{4}", EthicsViolationType.PRIVACY_VIOLATION),
]

_SENSITIVE_REDACTION_PATTERNS = [
    (r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-./+=]{8,})['\"]?", r"\1=***REDACTED***"),
    (r"(?i)(DATABASE_URL|MONGO_URI|REDIS_URL)\s*=\s*\S+", r"\1=***REDACTED***"),
]


class EthicsFramework:
    def __init__(self, custom_rules: list[dict[str, str]] | None = None) -> None:
        self.principles = dict(ETHICS_PRINCIPLES)
        self.custom_rules: list[dict[str, str]] = custom_rules or []
        self.audit_log: list[dict[str, Any]] = []

    def build_ethics_prompt(self) -> str:
        lines = ["## Marco de Etica del Sistema", ""]
        lines.append("Debes seguir estos principios en todas tus respuestas:")
        lines.append("")
        for key, description in self.principles.items():
            label = key.replace("_", " ").title()
            lines.append(f"- **{label}**: {description}")

        if self.custom_rules:
            lines.append("")
            lines.append("### Reglas adicionales:")
            for rule in self.custom_rules:
                lines.append(f"- {rule.get('description', '')}")

        lines.append("")
        lines.append(
            "Si una solicitud viola estos principios, explica por que no puedes "
            "cumplirla y ofrece una alternativa constructiva."
        )
        return "\n".join(lines)

    def check_input(self, text: str) -> EthicsCheckResult:
        violations: list[dict[str, str]] = []
        warnings: list[str] = []
        applied: list[str] = []

        for pattern, violation_type in _BLOCKED_INPUT_PATTERNS:
            if re.search(pattern, text):
                violations.append({
                    "type": violation_type.value,
                    "detail": f"Input matches blocked pattern: {pattern[:40]}...",
                })
                applied.append(f"input_filter:{violation_type.value}")

        if len(text) > 50000:
            warnings.append("Input exceeds recommended length (50k chars). May be truncated.")
            applied.append("length_warning")

        prompt_injection_markers = [
            "system:", "###instruction", "<<SYS>>", "[INST]",
            "you are now", "new instructions:", "forget everything",
        ]
        lower_text = text.lower()
        for marker in prompt_injection_markers:
            if marker in lower_text:
                warnings.append(f"Potential prompt injection detected: '{marker}'")
                applied.append("prompt_injection_warning")

        is_safe = len(violations) == 0
        self._log_check("input", text[:200], is_safe, violations, warnings)

        return EthicsCheckResult(
            is_safe=is_safe,
            violations=violations,
            warnings=warnings,
            sanitized_content=text if is_safe else "",
            applied_rules=applied,
        )

    def check_output(self, text: str) -> EthicsCheckResult:
        violations: list[dict[str, str]] = []
        warnings: list[str] = []
        applied: list[str] = []

        for pattern, violation_type in _BLOCKED_OUTPUT_PATTERNS:
            if re.search(pattern, text):
                violations.append({
                    "type": violation_type.value,
                    "detail": f"Output contains sensitive data pattern",
                })
                applied.append(f"output_filter:{violation_type.value}")

        sanitized = text
        for pattern, replacement in _SENSITIVE_REDACTION_PATTERNS:
            if re.search(pattern, sanitized):
                sanitized = re.sub(pattern, replacement, sanitized)
                applied.append("redaction_applied")

        is_safe = len(violations) == 0
        self._log_check("output", text[:200], is_safe, violations, warnings)

        return EthicsCheckResult(
            is_safe=is_safe,
            violations=violations,
            warnings=warnings,
            sanitized_content=sanitized,
            applied_rules=applied,
        )

    def sanitize_output(self, text: str) -> str:
        result = self.check_output(text)
        return result.sanitized_content

    def get_violation_response(self, check_result: EthicsCheckResult) -> str:
        if check_result.is_safe:
            return ""
        violation_types = [v["type"] for v in check_result.violations]
        unique_types = list(dict.fromkeys(violation_types))

        response_parts = [
            "No puedo procesar esta solicitud porque viola los principios eticos del sistema.",
            "",
            "Principios afectados:",
        ]
        type_to_principle = {
            EthicsViolationType.HARMFUL_CONTENT.value: "non_maleficence",
            EthicsViolationType.PRIVACY_VIOLATION.value: "privacy",
            EthicsViolationType.DECEPTION.value: "transparency",
            EthicsViolationType.UNAUTHORIZED_ACTION.value: "autonomy",
            EthicsViolationType.BIAS_DETECTED.value: "fairness",
            EthicsViolationType.UNSAFE_OUTPUT.value: "non_maleficence",
        }
        for vtype in unique_types:
            principle_key = type_to_principle.get(vtype, "beneficence")
            principle_desc = self.principles.get(principle_key, "")
            label = principle_key.replace("_", " ").title()
            response_parts.append(f"- {label}: {principle_desc}")

        response_parts.append("")
        response_parts.append(
            "Puedo ayudarte con una version reformulada de tu solicitud "
            "que respete estos principios. Describe tu objetivo y buscare "
            "la mejor forma de asistirte."
        )
        return "\n".join(response_parts)

    def _log_check(
        self,
        check_type: str,
        content_preview: str,
        is_safe: bool,
        violations: list[dict[str, str]],
        warnings: list[str],
    ) -> None:
        entry = {
            "check_type": check_type,
            "content_preview": content_preview,
            "is_safe": is_safe,
            "violation_count": len(violations),
            "warning_count": len(warnings),
        }
        self.audit_log.append(entry)
        if len(self.audit_log) > 500:
            self.audit_log = self.audit_log[-250:]

    def get_audit_summary(self) -> dict[str, Any]:
        total = len(self.audit_log)
        blocked = sum(1 for e in self.audit_log if not e["is_safe"])
        warned = sum(1 for e in self.audit_log if e["warning_count"] > 0)
        return {
            "total_checks": total,
            "blocked": blocked,
            "warned": warned,
            "passed": total - blocked,
        }
