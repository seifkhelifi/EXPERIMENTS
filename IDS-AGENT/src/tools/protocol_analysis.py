"""
IoT Protocol Analysis Tool
"""

from pydantic import BaseModel, Field
from typing import Literal, Dict
from database import SecurityDatabase


class ProtocolAnalysisAction(BaseModel):
    """Action to analyze IoT protocols and communication patterns"""

    action_type: Literal["protocol_analysis"] = "protocol_analysis"
    protocol: str = Field(description="Protocol to analyze (TCP, UDP, MQTT, etc.)")
    port: int = Field(description="Port number")
    flow_features: Dict = Field(description="Flow characteristics to analyze")
    reasoning: str = Field(description="Why protocol analysis is needed")


class ProtocolAnalysisTool:
    """Tool for IoT protocol analysis"""

    def __init__(self, db: SecurityDatabase):
        self.db = db

    def execute(self, action: ProtocolAnalysisAction) -> str:
        """Execute IoT protocol analysis"""
        protocol = action.protocol
        port = action.port
        features = action.flow_features

        analysis = f"Protocol Analysis - {protocol} on port {port}: "

        # Port-based protocol identification
        if port == 1883:  # MQTT
            analysis += "MQTT protocol detected. Standard IoT messaging protocol. "
        elif port == 80 or port == 443:
            analysis += "HTTP/HTTPS detected. Common for IoT device management. "
        elif port == 23 or port == 2323:
            analysis += (
                "Telnet detected. HIGH RISK - commonly exploited in IoT attacks. "
            )

        # Analyze flow characteristics
        if "duration" in features and features["duration"] < 0.1:
            analysis += "Very short duration - possible scanning or failed connection. "
        if "packet_count" in features and features["packet_count"] == 1:
            analysis += "Single packet - likely probe or scan attempt. "

        return analysis
