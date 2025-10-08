"""
Threat Intelligence Lookup Tool
"""

from pydantic import BaseModel, Field
from typing import Literal
from database import SecurityDatabase


class ThreatIntelLookupAction(BaseModel):
    """Action to check IP reputation and threat intelligence"""

    action_type: Literal["threat_intel"] = "threat_intel"
    ip_address: str = Field(description="IP address to check")
    reasoning: str = Field(description="Why this threat intel lookup is needed")


class ThreatIntelTool:
    """Tool for threat intelligence lookups"""

    def __init__(self, db: SecurityDatabase):
        self.db = db

    def execute(self, action: ThreatIntelLookupAction) -> str:
        """Execute threat intelligence lookup"""
        ip = action.ip_address
        intel = self.db.get_threat_intel(ip)

        if intel:
            return (
                f"Threat Intel for {ip}: "
                f"Reputation Score: {intel['reputation_score']}/10, "
                f"Malicious Indicators: {intel['malicious_indicators']}, "
                f"Last Seen: {intel['last_seen']}"
            )
        else:
            return f"Threat Intel for {ip}: No malicious indicators found, IP appears clean"
