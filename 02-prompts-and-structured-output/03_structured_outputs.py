"""
Module 02 - Prompts & Structured Output: 03_structured_outputs.py
Demonstrates enforcing Pydantic & TypedDict schemas via `.with_structured_output()`.
"""
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

class VulnerabilityReport(BaseModel):
    """Structured vulnerability triage record."""
    title: str = Field(description="Short summary of the security vulnerability")
    cve_id: Optional[str] = Field(None, description="CVE identifier if applicable")
    severity: str = Field(description="Severity rating: Critical, High, Medium, or Low")
    affected_packages: List[str] = Field(description="List of vulnerable packages and versions")
    remediation_steps: List[str] = Field(description="Specific actionable remediation commands or steps")

def main():
    print("=== Enforcing Pydantic Schema via .with_structured_output() ===")
    print("Pydantic JSON Schema:")
    print(VulnerabilityReport.model_json_schema())

    incident_log = """
    Alert: Remote Code Execution detected in auth-service container.
    Vulnerability matches CVE-2024-9981 in libssl version 1.1.1u.
    Immediate severity: Critical.
    Remediation: Upgrade libssl to 1.1.1w and patch the base Dockerfile image.
    """

    if not os.getenv("OPENAI_API_KEY"):
        print("\n[Mock Output] OPENAI_API_KEY not configured. Generating validated mock instance:")
        mock_result = VulnerabilityReport(
            title="Remote Code Execution in auth-service container",
            cve_id="CVE-2024-9981",
            severity="Critical",
            affected_packages=["libssl:1.1.1u"],
            remediation_steps=["Upgrade libssl to 1.1.1w", "Rebuild base Dockerfile image"]
        )
        print(mock_result.model_dump_json(indent=2))
        return

    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    structured_model = model.with_structured_output(VulnerabilityReport)
    
    result = structured_model.invoke(incident_log)
    print("\nValidated Output Object:")
    print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
