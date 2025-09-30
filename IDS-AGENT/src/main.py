import os
import json
import groq
from database import SecurityDatabase
from tool_registry import ToolRegistry
from models import IoTSecurityStep
from tools.respond import RespondAction

from dotenv import load_dotenv

load_dotenv()


class IoTSOCAgent:
    """IoT Security Operations Center Analyst ReAct Agent"""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = groq.Client(api_key=api_key)
        self.model = model
        self.conversation_history = []
        self.max_iterations = 6

        self.db = SecurityDatabase()
        self.tool_registry = ToolRegistry(self.db)

    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def get_structured_response(
        self, flow_data: str, context: str = ""
    ) -> IoTSecurityStep:
        """Get structured IoT security analysis response"""

        system_prompt = f"""You are an expert IoT SOC (Security Operations Center) analyst. You analyze network flows from IoT devices to detect security threats.

You have access to these specialized IoT security tools:
{self.tool_registry.get_tool_descriptions()}

IMPORTANT: You must respond with valid JSON matching this schema:
{{
  "thought": "your security analyst reasoning about this network flow",
  "action": {{
    "action_type": "threat_intel|device_profiler|protocol_analysis|attack_signature|flow_correlation|respond",
    "ip_address": "IP to check" (for threat_intel),
    "src_ip": "source IP" (for device_profiler/flow_correlation),
    "protocol": "protocol name" (for protocol_analysis),
    "port": port_number (for protocol_analysis),
    "flow_features": {{flow_characteristics}} (for protocol_analysis),
    "flow_pattern": {{pattern_data}} (for attack_signature),
    "time_window": "time_range" (for device_profiler),
    "time_range": "time_range" (for flow_correlation),
    "verdict": "benign|suspicious|malicious" (for respond),
    "confidence": confidence_score (for respond),
    "attack_type": "attack_type" (for respond),
    "response": "detailed_explanation" (for respond),
    "reasoning": "why you chose this action"
  }}
}}

Context from previous analysis: {context}

Focus on IoT-specific threats like device hijacking, botnet activity, unauthorized access, and protocol abuse."""

        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history,
            {"role": "user", "content": flow_data},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )

            response_text = response.choices[0].message.content
            print("🔍 Analysis JSON:", response_text)
            response_data = json.loads(response_text)

            return IoTSecurityStep.model_validate(response_data)

        except Exception as e:
            print(f"⚠️  Error in analysis: {e}")
            return IoTSecurityStep(
                thought="Encountered error during analysis, falling back to basic assessment",
                action=RespondAction(
                    verdict="suspicious",
                    confidence=0.5,
                    attack_type="unknown",
                    response="Analysis error occurred, manual review recommended",
                    reasoning="Fallback due to parsing error",
                ),
            )

    def analyze_flow(self, flow_csv_row: str) -> str:
        """Analyze a single flow from CIC-IoT dataset"""
        print(f"\n🔍 Analyzing IoT Flow: {flow_csv_row}")

        context = ""
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Security Analysis Step {iteration} ---")

            security_step = self.get_structured_response(flow_csv_row, context)

            print(f"💭 Analyst Thought: {security_step.thought}")
            print(f"🛠️  Action: {security_step.action.action_type}")
            print(f"📋 Reasoning: {security_step.action.reasoning}")

            result = self.tool_registry.execute_action(security_step.action)
            print(f"📊 Tool Result: {result}")

            step_summary = (
                f"Step {iteration} - Thought: {security_step.thought} | "
                f"Action: {security_step.action.action_type} | "
                f"Result: {result}"
            )
            self.add_to_history("assistant", step_summary)

            if isinstance(security_step.action, RespondAction):
                return result

            context += f"\n{step_summary}"
            flow_csv_row = result

        return "⚠️  Maximum analysis iterations reached. Manual review recommended."

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []


if __name__ == "__main__":

    api_key = os.getenv("API_KEY")
    iot_agent = IoTSOCAgent(api_key, "llama-3.3-70b-versatile")

    iot_flows = [
        "Network flow from IoT device 192.168.1.101 to server 10.0.0.2 on port 80 using TCP, duration 0.1 seconds, 1 forward packet, 0 backward packets, 64 bytes total. SYN flag set, no FIN flag.",
        # "MQTT communication from smart thermostat 192.168.1.10 to broker 192.168.1.1 on port 1883, duration 45.2 seconds, 12 forward packets, 11 backward packets, normal IoT sensor data exchange.",
        # "Suspicious activity from 172.16.1.50 to multiple IoT devices on port 23, duration 0.02 seconds, 1 packet each, RST flags, possible Telnet scanning attempt.",
    ]

    for i, flow in enumerate(iot_flows, 1):
        print("\n" + "=" * 70)
        print(f"🚨 ANALYZING IoT FLOW #{i}")
        print("=" * 70)

        final_verdict = iot_agent.analyze_flow(flow)

        print("\n" + "=" * 70)
        print(f"✅ FINAL SECURITY ASSESSMENT: {final_verdict}")
        print("=" * 70)

        # Reset for next flow analysis
        iot_agent.reset_conversation()
        print("\n" + "⏳ Moving to next flow analysis...")
