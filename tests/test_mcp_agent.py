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

    def describe_action(self, action):
        return {"action": action, "kind": "lane_keep", "target_speed_mps": 3.5}


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


def test_agent_previews_then_executes():
    ae = FakeAE()
    init_mcp_server(None, FakeWS(), ae)
    client = MCPDrivingClient()

    calls = []

    class FakeLLM:
        def generate_response(self, messages, tools=None):
            calls.append(len(messages))
            if len(calls) == 1:
                return FakeMsg([FakeTC("1", "get_world_state", "{}")])
            if len(calls) == 2:
                return FakeMsg(
                    [FakeTC("2", "preview_action", json.dumps({"action": "follow_lane"}))]
                )
            return FakeMsg(
                [FakeTC("3", "execute_action", json.dumps({"action": "follow_lane"}))]
            )

    action = DecisionMaker(FakeLLM(), client).run_step()
    assert action == "follow_lane"
    assert ae.last == "follow_lane"


def test_execute_without_preview_is_rejected():
    from simulation import step_context

    step_context.clear()
    ae = FakeAE()
    init_mcp_server(None, FakeWS(), ae)
    client = MCPDrivingClient()

    text = client.call_tool_sync("execute_action", {"action": "follow_lane"})
    result = json.loads(text)
    assert result.get("status") == "rejected"
    assert "preview_action" in result.get("message", "")
    assert ae.last is None


def test_stop_is_exempt_from_preview_gating():
    from simulation import step_context

    step_context.clear()
    ae = FakeAE()
    init_mcp_server(None, FakeWS(), ae)
    client = MCPDrivingClient()

    text = client.call_tool_sync("execute_action", {"action": "stop"})
    result = json.loads(text)
    assert result.get("status") == "success"
    assert ae.last == "stop"
