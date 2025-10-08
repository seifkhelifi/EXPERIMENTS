"""
Central registry for all security analysis tools
"""

from database import SecurityDatabase
from tools.threat_intel import ThreatIntelTool
from tools.device_profiler import DeviceProfilerTool
from tools.protocol_analysis import ProtocolAnalysisTool
from tools.attack_signature import AttackSignatureTool
from tools.flow_correlation import FlowCorrelationTool
from tools.respond import RespondTool


class ToolRegistry:
    """Registry for all security analysis tools"""

    def __init__(self, db: SecurityDatabase):
        self.db = db
        self.tools = {
            "threat_intel": ThreatIntelTool(db),
            "device_profiler": DeviceProfilerTool(db),
            "protocol_analysis": ProtocolAnalysisTool(db),
            "attack_signature": AttackSignatureTool(db),
            "flow_correlation": FlowCorrelationTool(db),
            "respond": RespondTool(db),
        }

    def execute_action(self, action) -> str:
        """Execute an action using the appropriate tool"""
        action_type = action.action_type

        if action_type == "threat_intel":
            return self.tools["threat_intel"].execute(action)
        elif action_type == "device_profiler":
            return self.tools["device_profiler"].execute(action)
        elif action_type == "protocol_analysis":
            return self.tools["protocol_analysis"].execute(action)
        elif action_type == "attack_signature":
            return self.tools["attack_signature"].execute(action)
        elif action_type == "flow_correlation":
            return self.tools["flow_correlation"].execute(action)
        elif action_type == "respond":
            return self.tools["respond"].execute(action)
        else:
            return f"Unknown action type: {action_type}"

    def get_tool_descriptions(self) -> str:
        """Get descriptions of all available tools"""
        return """  1. threat_intel: Check IP reputation against threat intelligence feeds
                    2. device_profiler: Analyze IoT device behavior patterns and baselines  
                    3. protocol_analysis: Deep analysis of IoT protocols (MQTT, CoAP, TCP, UDP)
                    4. attack_signature: Match flow patterns against known IoT attack signatures
                    5. flow_correlation: Correlate with recent network activity from same source
                    6. respond: Provide final security verdict with confidence score"""
