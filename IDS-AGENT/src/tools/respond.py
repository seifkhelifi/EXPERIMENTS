"""
Final Response Tool
"""

from pydantic import BaseModel, Field
from typing import Literal
from database import SecurityDatabase


class RespondAction(BaseModel):
    """Action to provide final security assessment"""

    action_type: Literal["respond"] = "respond"
    verdict: Literal["benign", "suspicious", "malicious"] = Field(
        description="Security verdict"
    )
    confidence: float = Field(description="Confidence score (0.0-1.0)")
    attack_type: str = Field(description="Type of attack if malicious")
    response: str = Field(description="Detailed explanation of the analysis")
    reasoning: str = Field(description="Why this is the final assessment")


class RespondTool:
    """Tool for providing final security assessment"""

    def __init__(self, db: SecurityDatabase):
        self.db = db

    def execute(self, action: RespondAction) -> str:
        """Execute final response"""
        return (
            f"🚨 SECURITY VERDICT: {action.verdict.upper()} | "
            f"Confidence: {action.confidence:.2f} | "
            f"Attack Type: {action.attack_type} | "
            f"Analysis: {action.response}"
        )
