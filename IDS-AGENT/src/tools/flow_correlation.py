"""
Flow Correlation Tool
"""

from pydantic import BaseModel, Field
from typing import Literal
from database import SecurityDatabase


class FlowCorrelationAction(BaseModel):
    """Action to correlate current flow with recent network activity"""

    action_type: Literal["flow_correlation"] = "flow_correlation"
    src_ip: str = Field(description="Source IP for correlation")
    time_range: str = Field(description="Time range for correlation")
    reasoning: str = Field(description="Why flow correlation is needed")


class FlowCorrelationTool:
    """Tool for flow correlation analysis"""

    def __init__(self, db: SecurityDatabase):
        self.db = db

    def execute(self, action: FlowCorrelationAction) -> str:
        """Execute flow correlation analysis"""
        ip = action.src_ip
        flows = self.db.get_recent_flows(ip)

        if flows:
            return (
                f"Recent Activity from {ip}: "
                f"{len(flows)} flows in past hour. "
                f"Pattern: {flows[:3]}... (showing first 3)"
            )
        else:
            return f"No recent correlated flows found for {ip}"
