from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource



def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:discord:group:c1:u1",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._smart_model_routing = {"enabled": False}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "gemma reply",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    return runner


class TestGemmaRouting:
    def test_ai_gemma_channel_uses_delegation_runtime(self, monkeypatch):
        from gateway.run import GatewayRunner

        runner = SimpleNamespace(_smart_model_routing={"enabled": False})
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="c1",
            chat_name="EchoChamber / #ai-gemma",
            chat_type="group",
        )

        monkeypatch.setattr(
            "gateway.run._load_gateway_config",
            lambda: {
                "delegation": {
                    "model": "gemma-4-e4b-it",
                    "provider": "custom",
                    "base_url": "http://100.96.53.33:1234/v1",
                    "api_key": "lmstudio-token",
                }
            },
        )

        bound = GatewayRunner._resolve_turn_agent_config.__get__(runner)
        result = bound(
            "hello from gemma channel",
            "gpt-5.4",
            {
                "api_key": "primary-key",
                "base_url": None,
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "command": None,
                "args": [],
                "credential_pool": None,
            },
            source=source,
        )

        assert result["model"] == "gemma-4-e4b-it"
        assert result["runtime"]["provider"] == "custom"
        assert result["runtime"]["base_url"] == "http://100.96.53.33:1234/v1"
        assert result["runtime"]["api_key"] == "lmstudio-token"

    @pytest.mark.asyncio
    async def test_gemma_command_forces_gemma_route_and_forwards_args(self, monkeypatch):
        runner = _make_runner()
        event = MessageEvent(
            text="/gemma Summarize this thread",
            source=SessionSource(
                platform=Platform.DISCORD,
                user_id="u1",
                chat_id="c1",
                chat_name="EchoChamber / #general",
                user_name="tester",
                chat_type="group",
            ),
            message_id="m1",
        )

        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {
                "api_key": "primary-key",
                "base_url": None,
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "command": None,
                "args": [],
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(
            "gateway.run._load_gateway_config",
            lambda: {
                "delegation": {
                    "model": "gemma-4-e4b-it",
                    "provider": "custom",
                    "base_url": "http://100.96.53.33:1234/v1",
                    "api_key": "lmstudio-token",
                }
            },
        )
        monkeypatch.setattr(
            "agent.model_metadata.get_model_context_length",
            lambda *_args, **_kwargs: 100_000,
        )

        result = await runner._handle_message(event)

        assert result == "gemma reply"
        assert runner._run_agent.call_args.kwargs["message"] == "Summarize this thread"
        assert runner._run_agent.call_args.kwargs["route_override"] == {"name": "gemma"}
