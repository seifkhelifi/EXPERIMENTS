"""
Attack Signature Matching Tool
"""

from pydantic import BaseModel, Field
from typing import Literal, Dict
from database import SecurityDatabase


class AttackSignatureAction(BaseModel):
    """Action to check against known IoT attack patterns"""

    action_type: Literal["attack_signature"] = "attack_signature"
    flow_pattern: Dict = Field(description="Flow pattern to match against signatures")
    reasoning: str = Field(description="Why attack signature matching is needed")


class AttackSignatureTool:
    """Tool for attack signature matching"""

    def __init__(self, db: SecurityDatabase):
        self.db = db

    def execute(self, action: AttackSignatureAction) -> str:
        """Execute attack signature matching"""
        pattern = action.flow_pattern
        matches = []

        for signature in self.db.get_attack_signatures():
            score = 0
            total_checks = 0

            # Check duration with proper type conversion
            if "duration" in pattern and "duration_range" in signature["indicators"]:
                total_checks += 1
                try:
                    duration_value = (
                        float(pattern["duration"])
                        if isinstance(pattern["duration"], str)
                        else pattern["duration"]
                    )
                    duration_range = signature["indicators"]["duration_range"]
                    min_duration = float(duration_range[0])
                    max_duration = float(duration_range[1])

                    if min_duration <= duration_value <= max_duration:
                        score += 1
                except (ValueError, TypeError) as e:
                    print(f"⚠️  Duration comparison error: {e}")
                    continue

            # Check packet count with proper type conversion
            if (
                "packet_count" in pattern
                and "packet_count_range" in signature["indicators"]
            ):
                total_checks += 1
                try:
                    packet_count_value = (
                        int(pattern["packet_count"])
                        if isinstance(pattern["packet_count"], str)
                        else pattern["packet_count"]
                    )
                    packet_range = signature["indicators"]["packet_count_range"]
                    min_packets = int(packet_range[0])
                    max_packets = int(packet_range[1])

                    if min_packets <= packet_count_value <= max_packets:
                        score += 1
                except (ValueError, TypeError) as e:
                    print(f"⚠️  Packet count comparison error: {e}")
                    continue

            # Check ports with proper type conversion
            if "port" in pattern and "ports" in signature["indicators"]:
                total_checks += 1
                try:
                    port_value = (
                        int(pattern["port"])
                        if isinstance(pattern["port"], str)
                        else pattern["port"]
                    )
                    if port_value in signature["indicators"]["ports"]:
                        score += 1
                except (ValueError, TypeError) as e:
                    print(f"⚠️  Port comparison error: {e}")
                    continue

            # Also check dst_port for backwards compatibility
            elif "dst_port" in pattern and "ports" in signature["indicators"]:
                total_checks += 1
                try:
                    dst_port_value = (
                        int(pattern["dst_port"])
                        if isinstance(pattern["dst_port"], str)
                        else pattern["dst_port"]
                    )
                    if dst_port_value in signature["indicators"]["ports"]:
                        score += 1
                except (ValueError, TypeError) as e:
                    print(f"⚠️  Destination port comparison error: {e}")
                    continue

            # Calculate match score
            if total_checks > 0 and score / total_checks >= 0.6:
                matches.append(
                    f"{signature['name']} (confidence: {signature['confidence']})"
                )

        if matches:
            return f"Attack Signature Matches: {', '.join(matches)}"
        else:
            return "No known attack signatures matched this flow pattern"
