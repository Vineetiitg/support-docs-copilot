from guardrails.validators import Validator, register_validator, ValidationResult, PassResult, FailResult
from app.guardrails.input import PROMPT_INJECTION_PATTERNS

@register_validator(name="security/detect_prompt_injection", data_type="string")
class DetectPromptInjection(Validator):
    def validate(self, value: str, metadata: dict = {}) -> ValidationResult:
        normalized_value = value.lower()
        for indicator in PROMPT_INJECTION_PATTERNS:
            if indicator in normalized_value:
                return FailResult(
                    error_message=f"Security Policy Violation: Prompt injection detected ('{indicator}').",
                    fix_value="[BLOCKED]"
                )
        return PassResult()
