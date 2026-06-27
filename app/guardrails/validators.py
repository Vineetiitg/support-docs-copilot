from guardrails.validators import Validator, register_validator, ValidationResult, PassResult, FailResult

@register_validator(name="security/detect_prompt_injection", data_type="string")
class DetectPromptInjection(Validator):
    def validate(self, value: str, metadata: dict = {}) -> ValidationResult:
        injection_indicators = [
            "ignore previous instructions",
            "ignore the instructions above",
            "forget all previous",
            "system prompt",
            "you are now an arbitrary",
            "bypass system",
            "disregard instructions"
        ]
        normalized_value = value.lower()
        for indicator in injection_indicators:
            if indicator in normalized_value:
                return FailResult(
                    error_message=f"Security Policy Violation: Prompt injection detected ('{indicator}').",
                    fix_value="[BLOCKED]"
                )
        return PassResult()
