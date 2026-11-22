# Architecture Guidelines for Agentic Systems

## Core Tenets
1. **Deterministic Guardrails**: LLMs must be bounded by structured schemas and type-validated tools.
2. **State Isolation**: Agent graphs must maintain explicit, serializable state checkpoints.
3. **Observability**: Every tool invocation and model reasoning cycle must emit traceable spans.

## Security Policies
- Never execute raw code without human-in-the-loop validation or isolated sandboxes.
- Redact PII before propagating data to cloud LLM providers.
