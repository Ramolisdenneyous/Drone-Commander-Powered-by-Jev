"""Jev menu, shadow, radio, and fallback checks against a real battle."""

from __future__ import annotations

import json

import pytest

from app.agents import provider
from app.agents.orchestration import run_agent_activation
from app.config import settings
from app.content.loader import get_catalog, load_catalog
from app.domain.hex import Hex
from app.engine.battle import create_battle
from app.engine.options import build_options, fallback_select
from app.engine.state import ActionPool, ActivationState


def _prep(definition_id: str) -> dict:
    return {
        "mission_id": "freestyle_vs_15",
        "map_id": "vs_middle_east_50",
        "point_cap": 15,
        "avatar": "male",
        "ram_abilities": ["defense_matrix"],
        "army": [{"definition_id": definition_id, "count": 1}],
    }


@pytest.fixture
def jev_live(monkeypatch):
    load_catalog(force=True)
    monkeypatch.setattr(settings, "llm_external_enabled", True)
    monkeypatch.setattr(settings, "tactical_provider", "jev")
    monkeypatch.setattr(settings, "typesafe_api_key", "test-key")
    monkeypatch.setattr(settings, "tactical_shadow", False)
    monkeypatch.setattr(settings, "radio_provider", "auto")
    monkeypatch.setattr(settings, "openai_api_key", "should-not-call")
    monkeypatch.setattr(settings, "jev_min_confidence", 0.0)
    monkeypatch.setattr(settings, "jev_min_margin", 0.0)


def _activate(battle, definition_id):
    unit = next(u for u in battle.units.values() if u.definition_id == definition_id and u.alive)
    battle.activation = ActivationState(f"act-{definition_id}", unit.unit_instance_id, ActionPool())
    return unit


def _spy(monkeypatch, choose):
    captured: dict = {}

    def fake_select(**kwargs):
        captured["payload"] = kwargs["user_payload"]
        captured["ids"] = list(kwargs["offered_option_ids"])
        return {
            "option_id": choose(captured["ids"], captured["payload"]),
            "fallback_policy": "hold",
            "reason": "test",
            "raw": {
                "provider": "typesafe",
                "model": "jev-test",
                "confidence": 0.22,
                "margin": 0.04,
                "usage": {},
            },
        }

    monkeypatch.setattr(provider, "select_tactical_option", fake_select)
    return captured


def _events(battle, event_type: str) -> list[dict]:
    return [event for event in battle.events if event["type"] == event_type]


def test_engineer_suicide_and_objective_payloads_stay_bounded(monkeypatch, jev_live):
    cases = [
        ("friendly_combat_engineers", "engineer_mine_then_guns"),
        ("friendly_one_way_drone", "disposable_bomber_hunt"),
        ("friendly_infantry_squad", "temple_control"),
    ]
    for definition_id, marker in cases:
        battle = create_battle(f"jev-{definition_id}", _prep(definition_id), seed=11)
        _activate(battle, definition_id)
        captured = _spy(monkeypatch, lambda ids, _payload: ids[0])
        run_agent_activation(battle, db=None)
        blob = json.dumps(captured["payload"])
        assert marker in blob
        assert "raw_text" not in blob
        assert captured["ids"]
        assert _events(battle, "agent_choice_received")


def test_empty_direct_drone_offers_resupply_instead_of_fire(monkeypatch, jev_live):
    battle = create_battle("jev-resupply", _prep("friendly_direct_attack_drone"), seed=12)
    unit = _activate(battle, "friendly_direct_attack_drone")
    for weapon_id in list(unit.weapons):
        unit.ammo[weapon_id] = 0
    unit.position = Hex(battle.width // 2, battle.height // 2)
    for model in unit.models:
        model.position = unit.position
    captured = _spy(monkeypatch, lambda ids, _payload: ids[0])
    run_agent_activation(battle, db=None)
    subroutines = {item.get("subroutine") for item in captured["payload"]["legal_options"]}
    assert "attack" not in subroutines
    assert subroutines & {"move", "return_to_signal", "return_to_resupply"}


def test_red_commander_first_choice_includes_signal_rules(monkeypatch, jev_live):
    battle = create_battle("jev-commander", _prep("friendly_infantry_squad"), seed=13)
    battle.control_phase = None
    _activate(battle, "opposition_commander")
    captured = _spy(monkeypatch, lambda ids, _payload: ids[0])
    run_agent_activation(battle, db=None)
    blob = json.dumps(captured["payload"]).lower()
    assert "signal" in blob
    assert captured["ids"]
    assert _events(battle, "agent_choice_received") or _events(battle, "agent_fallback_used")


def test_stale_option_falls_back_without_committing_it(monkeypatch, jev_live):
    battle = create_battle("jev-stale", _prep("friendly_infantry_squad"), seed=14)
    _activate(battle, "opposition_line_cell")
    _spy(monkeypatch, lambda _ids, _payload: "not-a-real-option")
    before = battle.state_version
    run_agent_activation(battle, db=None)
    assert battle.state_version > before
    assert _events(battle, "agent_fallback_used") or any(
        "stale_option_rebuilt" in json.dumps(event) for event in battle.events
    )


def test_shadow_logs_jev_and_commits_fallback(monkeypatch, jev_live):
    monkeypatch.setattr(settings, "tactical_shadow", True)
    battle = create_battle("jev-shadow", _prep("friendly_infantry_squad"), seed=15)
    unit = _activate(battle, "opposition_line_cell")
    options = build_options(get_catalog(), battle, unit)
    battle.activation.options = options
    expected = fallback_select(options, unit, battle)

    def choose(ids, _payload):
        return next((option_id for option_id in reversed(ids) if option_id != expected), ids[0])

    _spy(monkeypatch, choose)
    run_agent_activation(battle, db=None)
    observed = _events(battle, "agent_shadow_observed")
    assert observed
    payload = observed[-1]["payload"]
    assert payload["committed_option_id"] == expected
    assert payload["agreement"] == (payload["jev_option_id"] == expected)
    assert payload["confidence"] == 0.22
    assert "agent_choice_received" not in {event["type"] for event in battle.events}


def test_friendly_radio_stays_on_the_template(monkeypatch, jev_live):
    battle = create_battle("jev-radio", _prep("friendly_combat_engineers"), seed=16)
    unit = _activate(battle, "friendly_combat_engineers")
    _spy(monkeypatch, lambda ids, _payload: ids[0])
    for _ in range(4):
        if battle.activation is None or battle.activation.actor_id != unit.unit_instance_id:
            break
        run_agent_activation(battle, db=None)
    texts = [entry.get("text", "") for entry in battle.communications]
    assert any(text.endswith("complete.") for text in texts)
    assert all("should-not-call" not in text for text in texts)
