"""
Module 09 - MCP and Middleware: 02_pii_redaction.py
Intercepting inputs/outputs with PII redaction middleware.
"""
import re
from typing import Tuple

class PIIRedactionMiddleware:
    """Regex-based PII redaction guardrail to sanitize sensitive tokens before LLM ingestion."""

    EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"
    API_KEY_PATTERN = r"(?i)sk-[a-zA-Z0-9]{20,}"

    @classmethod
    def redact(cls, text: str) -> Tuple[str, dict]:
        redactions = {}

        # Mask Emails
        def email_repl(match):
            placeholder = f"[REDACTED_EMAIL_{len(redactions)}]"
            redactions[placeholder] = match.group(0)
            return placeholder
        sanitized = re.sub(cls.EMAIL_PATTERN, email_repl, text)

        # Mask SSNs
        def ssn_repl(match):
            placeholder = f"[REDACTED_SSN_{len(redactions)}]"
            redactions[placeholder] = match.group(0)
            return placeholder
        sanitized = re.sub(cls.SSN_PATTERN, ssn_repl, sanitized)

        # Mask API Keys
        def key_repl(match):
            placeholder = f"[REDACTED_APIKEY_{len(redactions)}]"
            redactions[placeholder] = match.group(0)
            return placeholder
        sanitized = re.sub(cls.API_KEY_PATTERN, key_repl, sanitized)

        return sanitized, redactions

def main():
    print("=== PII Redaction Guardrail Middleware ===")

    raw_user_input = """
    Hello support team, my email is john.doe@enterprise.corp and my SSN is 000-12-3456.
    I accidentally committed my API key sk-91823719827391827391 to our public repository!
    """

    print("Original Raw Input:")
    print(raw_user_input.strip())

    sanitized_input, vault = PIIRedactionMiddleware.redact(raw_user_input)

    print("\nSanitized Input (Safe for external LLM ingestion):")
    print(sanitized_input.strip())

    print(f"\nSecure Vault Map (Stored locally for unmasking if required):")
    for placeholder, original in vault.items():
        print(f"  {placeholder} -> {original}")

if __name__ == "__main__":
    main()
