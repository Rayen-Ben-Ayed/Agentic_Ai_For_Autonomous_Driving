import json
from unittest.mock import MagicMock

from mcp_interface.server import init_mcp_server
from mcp_interface.client import MCPDrivingClient
from agent.decision_maker import DecisionMaker


class FakeWS:
    def get_state(self):
        return {"obstacle_ahead": False}


class FakeAE:
    def __init__(self):
        self.last = None

    def execute_action(self, action):
        self.last = action
        return True


def test_agent_uses_mcp_tools():
    ae = FakeAE()
    init_mcp_server(None, FakeWS(), ae)
    client = MCPDrivingClient()

    class FakeMsg:
        def __init__(self, tool_calls=None, content=None):
            self.tool_calls = tool_calls or []
            self.content = content

    class FakeTC:
        def __init__(self, tc_id, name, args):
            self.id = tc_id
            self.function = MagicMock()
            self.function.name = name
            self.function.arguments = args

    calls = []

    class FakeLLM:
        def generate_response(self, messages, tools=None):
            calls.append(len(messages))
            if len(calls) == 1:
                return FakeMsg([FakeTC("1", "get_world_state", "{}")])
            return FakeMsg(
                [FakeTC("2", "execute_action", json.dumps({"action": "follow_lane"}))]
            )

    action = DecisionMaker(FakeLLM(), client).run_step()
    assert action == "follow_lane"
    assert ae.last == "follow_lane"
