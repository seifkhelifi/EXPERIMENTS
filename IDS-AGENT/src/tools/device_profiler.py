"""
IoT Device Profiler Tool
"""

from pydantic import BaseModel, Field
from typing import Literal
from database import SecurityDatabase


class DeviceProfilerAction(BaseModel):
    """Action to analyze IoT device behavior and create baseline"""

    action_type: Literal["device_profiler"] = "device_profiler"
    src_ip: str = Field(description="Source IP to profile")
    time_window: str = Field(description="Time window for analysis (e.g., '1h', '24h')")
    reasoning: str = Field(description="Why device profiling is needed")


class DeviceProfilerTool:
    """Tool for IoT device behavior profiling"""

    def __init__(self, db: SecurityDatabase):
        self.db = db

    def execute(self, action: DeviceProfilerAction) -> str:
        """Execute IoT device behavior profiling"""
        ip = action.src_ip
        profile = self.db.get_device_profile(ip)

        if profile:
            return (
                f"Device Profile for {ip}: "
                f"Type: {profile['device_type']}, "
                f"Manufacturer: {profile['manufacturer']}, "
                f"Normal Patterns: {profile['normal_patterns']}"
            )
        else:
            return (
                f"Device Profile for {ip}: Unknown device, no baseline available. "
                f"Recommended to establish behavioral baseline."
            )
