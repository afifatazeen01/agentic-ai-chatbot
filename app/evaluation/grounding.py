from pydantic import BaseModel, Field
from typing import List

class GroundingVerification(BaseModel):
    """Structured output for grounding verification."""
    grounded: bool = Field(description="True if the answer is fully supported by the retrieved context, False otherwise")
    score: float = Field(description="A score from 0.0 to 1.0 indicating how well the answer is grounded in the context")
    unsupported_claims: List[str] = Field(description="List of claims in the answer that are not supported by the context. Empty if fully grounded.")

GROUNDING_VERIFICATION_PROMPT = """You are a strict fact-checker. 
Your job is to verify if the GENERATED_ANSWER is completely supported by the RETRIEVED_CONTEXT.

Rules:
1. Every claim in the GENERATED_ANSWER must be traceable to the RETRIEVED_CONTEXT.
2. If the GENERATED_ANSWER contains information not present in the RETRIEVED_CONTEXT, it is not grounded.
3. If the answer explicitly states that the information is not available in the context (a refusal), it IS considered grounded.

USER QUERY:
{query}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}
"""
