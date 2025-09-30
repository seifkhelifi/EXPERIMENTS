"""
Security databases for IoT threat intelligence and device profiling
"""

from typing import Dict, List


class SecurityDatabase:
    """Centralized security database for IoT analysis"""

    def __init__(self):
        self.threat_intel_db = self._init_threat_intel_db()
        self.device_profiles_db = self._init_device_profiles_db()
        self.attack_signatures_db = self._init_attack_signatures_db()
        self.recent_flows_db = self._init_recent_flows_db()

    def _init_threat_intel_db(self) -> Dict:
        """Initialize simulated threat intelligence database"""
        return {
            "192.168.1.101": {
                "reputation_score": 8.5,
                "malicious_indicators": ["botnet_c2", "iot_malware"],
                "last_seen": "2024-06-05",
                "threat_feeds": ["abuse_ip", "malware_db"],
            },
            "10.0.0.15": {
                "reputation_score": 2.1,
                "malicious_indicators": [],
                "last_seen": "2024-06-06",
                "threat_feeds": [],
            },
            "172.16.1.50": {
                "reputation_score": 9.2,
                "malicious_indicators": ["ddos_source", "scanning_activity"],
                "last_seen": "2024-06-06",
                "threat_feeds": ["security_vendors", "honeypot_data"],
            },
        }

    def _init_device_profiles_db(self) -> Dict:
        """Initialize IoT device behavioral profiles"""
        return {
            "192.168.1.10": {
                "device_type": "smart_thermostat",
                "manufacturer": "Nest",
                "normal_patterns": {
                    "avg_packet_count": 12,
                    "avg_duration": 45.2,
                    "typical_ports": [80, 443, 8080],
                    "communication_frequency": "every_5_minutes",
                    "protocol_distribution": {"TCP": 0.8, "UDP": 0.2},
                },
            },
            "192.168.1.20": {
                "device_type": "security_camera",
                "manufacturer": "Ring",
                "normal_patterns": {
                    "avg_packet_count": 150,
                    "avg_duration": 300.0,
                    "typical_ports": [443, 1935],
                    "communication_frequency": "continuous_streaming",
                    "protocol_distribution": {"TCP": 0.95, "UDP": 0.05},
                },
            },
        }

    def _init_attack_signatures_db(self) -> List[Dict]:
        """Initialize IoT attack signature database"""
        return [
            {
                "name": "IoT_DDoS_Mirai",
                "indicators": {
                    "duration_range": [0.01, 0.5],
                    "packet_count_range": [1, 5],
                    "response_rate": 0.0,
                    "ports": [23, 2323, 80, 8080],
                },
                "confidence": 0.9,
            },
            {
                "name": "IoT_Port_Scan",
                "indicators": {
                    "duration_range": [0.001, 0.1],
                    "packet_count_range": [1, 3],
                    "response_rate": 0.0,
                    "sequential_ports": True,
                },
                "confidence": 0.85,
            },
            {
                "name": "MQTT_Hijack",
                "indicators": {
                    "ports": [1883, 8883],
                    "unusual_topic_patterns": True,
                    "high_frequency": True,
                },
                "confidence": 0.88,
            },
        ]

    def _init_recent_flows_db(self) -> Dict:
        """Initialize recent network flows for correlation"""
        return {
            "192.168.1.101": [
                {
                    "timestamp": "2024-06-06 10:15:30",
                    "dst_port": 80,
                    "duration": 0.1,
                    "packets": 1,
                },
                {
                    "timestamp": "2024-06-06 10:15:35",
                    "dst_port": 443,
                    "duration": 0.1,
                    "packets": 1,
                },
                {
                    "timestamp": "2024-06-06 10:15:40",
                    "dst_port": 8080,
                    "duration": 0.05,
                    "packets": 1,
                },
                {
                    "timestamp": "2024-06-06 10:15:45",
                    "dst_port": 22,
                    "duration": 0.02,
                    "packets": 1,
                },
            ]
        }

    def get_threat_intel(self, ip: str) -> Dict:
        """Get threat intelligence for an IP"""
        return self.threat_intel_db.get(ip)

    def get_device_profile(self, ip: str) -> Dict:
        """Get device profile for an IP"""
        return self.device_profiles_db.get(ip)

    def get_attack_signatures(self) -> List[Dict]:
        """Get all attack signatures"""
        return self.attack_signatures_db

    def get_recent_flows(self, ip: str) -> List[Dict]:
        """Get recent flows for an IP"""
        return self.recent_flows_db.get(ip)
