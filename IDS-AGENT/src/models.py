from pydantic import BaseModel, Field
from typing import Literal, Union, Dict


class ThreatIntelLookupAction(BaseModel):
    """Action to check IP reputation and threat intelligence"""

    action_type: Literal["threat_intel"] = "threat_intel"
    ip_address: str = Field(description="IP address to check")
    reasoning: str = Field(description="Why this threat intel lookup is needed")


class DeviceProfilerAction(BaseModel):
    """Action to analyze IoT device behavior and create baseline"""

    action_type: Literal["device_profiler"] = "device_profiler"
    src_ip: str = Field(description="Source IP to profile")
    time_window: str = Field(description="Time window for analysis (e.g., '1h', '24h')")
    reasoning: str = Field(description="Why device profiling is needed")


class ProtocolAnalysisAction(BaseModel):
    """Action to analyze IoT protocols and communication patterns"""

    action_type: Literal["protocol_analysis"] = "protocol_analysis"
    protocol: str = Field(description="Protocol to analyze (TCP, UDP, MQTT, etc.)")
    port: int = Field(description="Port number")
    flow_features: Dict = Field(description="Flow characteristics to analyze")
    reasoning: str = Field(description="Why protocol analysis is needed")


class AttackSignatureAction(BaseModel):
    """Action to check against known IoT attack patterns"""

    action_type: Literal["attack_signature"] = "attack_signature"
    flow_pattern: Dict = Field(description="Flow pattern to match against signatures")
    reasoning: str = Field(description="Why attack signature matching is needed")


class FlowCorrelationAction(BaseModel):
    """Action to correlate current flow with recent network activity"""

    action_type: Literal["flow_correlation"] = "flow_correlation"
    src_ip: str = Field(description="Source IP for correlation")
    time_range: str = Field(description="Time range for correlation")
    reasoning: str = Field(description="Why flow correlation is needed")


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


class IoTSecurityStep(BaseModel):
    """A single step in IoT security analysis"""

    thought: str = Field(description="Security analyst reasoning about the flow")
    action: Union[
        ThreatIntelLookupAction,
        DeviceProfilerAction,
        ProtocolAnalysisAction,
        AttackSignatureAction,
        FlowCorrelationAction,
        RespondAction,
    ] = Field(
        description="The security analysis action to take", discriminator="action_type"
    )
