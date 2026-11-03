"""LLM provider adapters for tactical option selection and radio."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ..config import settings

logger = logging.getLogger("uvicorn.error")

_SAFE_PREVIEW_KEYS = {
    "blast_radius",
    "enemies_in_blast",
    "friendlies_in_blast",
    "enemy_models_in_blast",
    "friendly_models_in_blast",
    "friendly_fire",
    "under_enemy",
    "in_own_deploy",
    "near_objective",
    "on_approach",
    "center_dist",
    "risk_tags",
    "ability_id",
    "target_unit_id",
    "weapon_id",
    "dash",
    "closes_on_objective",
    "enters_objective",
    "closes_on_resupply",
    "enters_resupply",
}
_jev_catalog: dict[str, Any] = {"ok": False, "models": []}

SELECT_TOOL = {
    "type": "function",
    "function": {
        "name": "select_tactical_option",
        "description": "Select exactly one offered tactical option.",
        "parameters": {
            "type": "object",
            "properties": {
                "activation_id": {"type": "string"},
                "option_id": {"type": "string"},
                "fallback_policy": {
                    "type": "string",
                    "enum": ["next_best_target", "nearest_cover", "continue_objective", "return_to_signal", "hold"],
                },
                "reason": {"type": "string"},
            },
            "required": ["activation_id", "option_id", "fallback_policy"],
        },
    },
}


class ProviderError(Exception):
    pass


def select_tactical_option(
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    activation_id: str,
    offered_option_ids: list[str],
) -> dict[str, Any]:
    """Select a legal option through the configured tactical provider."""
    provider = settings.tactical_provider.strip().lower()
    if provider == "jev":
        return _select_jev_tactical_option(
            user_payload=user_payload,
            activation_id=activation_id,
            offered_option_ids=offered_option_ids,
        )
    if provider != "openai":
        raise ProviderError(f"Unknown tactical provider '{provider}'")
    return _select_openai_tactical_option(
        system_prompt=system_prompt,
        user_payload=user_payload,
        activation_id=activation_id,
        offered_option_ids=offered_option_ids,
    )


def tactical_provider_ready() -> bool:
    """Whether a configured external tactical provider has the required local key."""
    if not settings.llm_external_enabled:
        return False
    provider = settings.tactical_provider.strip().lower()
    if provider == "jev":
        return bool(settings.typesafe_api_key)
    return provider == "openai" and bool(settings.openai_api_key)


def tactical_shadow_enabled() -> bool:
    """Jev may observe a live menu, but the deterministic fallback commits the action."""
    return bool(settings.tactical_shadow) and settings.tactical_provider.strip().lower() == "jev"


def radio_provider_name() -> str:
    """Jev does not write prose. Auto mode keeps those tests on the template radio."""
    mode = settings.radio_provider.strip().lower()
    if mode == "template":
        return "template"
    openai_ready = bool(settings.llm_external_enabled and settings.openai_api_key)
    if mode == "openai":
        return "openai" if openai_ready else "template"
    if settings.tactical_provider.strip().lower() == "jev":
        return "template"
    return "openai" if openai_ready else "template"


def discovered_jev_models() -> dict[str, Any]:
    return {"ok": bool(_jev_catalog.get("ok")), "models": list(_jev_catalog.get("models") or [])}


def refresh_jev_model_catalog() -> dict[str, Any]:
    """Record TypeSafe model ids. Never stores or logs the API key."""
    global _jev_catalog
    if settings.tactical_provider.strip().lower() != "jev" or not settings.typesafe_api_key:
        _jev_catalog = {"ok": False, "models": []}
        return discovered_jev_models()
    url = settings.typesafe_base_url.rstrip("/") + "/v1/models"
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(url, headers={"Authorization": f"Bearer {settings.typesafe_api_key}"})
            if resp.status_code >= 400:
                raise ProviderError(f"TypeSafe models {resp.status_code}")
            data = resp.json()
        names = [
            str(item.get("name"))
            for item in (data.get("models") or [])
            if isinstance(item, dict) and item.get("name")
        ]
        _jev_catalog = {"ok": True, "models": names}
    except Exception as exc:
        logger.warning("TypeSafe model discovery failed: %s", type(exc).__name__)
        _jev_catalog = {"ok": False, "models": []}
    return discovered_jev_models()


def _select_openai_tactical_option(
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    activation_id: str,
    offered_option_ids: list[str],
) -> dict[str, Any]:
    """Call OpenAI gpt-5.6-luna to pick an option_id via tool call."""
    if not settings.llm_external_enabled or not settings.openai_api_key:
        raise ProviderError("LLM disabled or missing API key")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
    ]
    body = {
        "model": settings.llm_model_tactical,
        "messages": messages,
        "tools": [SELECT_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "select_tactical_option"}},
        # Luna requires reasoning_effort=none for function tools on chat.completions
        "reasoning_effort": "none",
        "max_completion_tokens": 300,
    }
    url = settings.openai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=settings.llm_decision_timeout_sec) as client:
            resp = client.post(url, headers=headers, json=body)
            if resp.status_code >= 400:
                raise ProviderError(f"{resp.status_code}: {resp.text[:400]}")
            data = resp.json()
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(str(exc)) from exc

    try:
        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            raise ProviderError("No tool call in response")
        args = json.loads(tool_calls[0]["function"]["arguments"])
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"Malformed tool response: {exc}") from exc

    option_id = args.get("option_id")
    if option_id not in offered_option_ids:
        raise ProviderError(f"Unoffered option_id {option_id}")
    if args.get("activation_id") != activation_id:
        raise ProviderError("activation_id mismatch")
    return {
        "option_id": option_id,
        "fallback_policy": args.get("fallback_policy", "hold"),
        "reason": args.get("reason", ""),
        "raw": {"model": settings.llm_model_tactical, "usage": data.get("usage")},
    }


def _listed_options(user_payload: dict[str, Any]) -> list[dict[str, Any]]:
    listed = user_payload.get("legal_options") or user_payload.get("offered_options") or []
    return [item for item in listed if isinstance(item, dict) and item.get("option_id")]


def _criteria_blob(option: dict[str, Any]) -> str:
    preview = option.get("preview") if isinstance(option.get("preview"), dict) else {}
    compact_preview = {}
    for key in _SAFE_PREVIEW_KEYS:
        if key in option:
            compact_preview[key] = option[key]
        elif key in preview:
            compact_preview[key] = preview[key]
    return json.dumps(
        {
            "subroutine": option.get("subroutine"),
            "label": option.get("label"),
            "action_cost": option.get("action_cost"),
            "preview": compact_preview,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _jev_state(user_payload: dict[str, Any]) -> dict[str, Any]:
    """Build compact, trusted decision state; player free text is intentionally omitted."""
    directives = []
    for directive in user_payload.get("directives") or []:
        if not isinstance(directive, dict):
            continue
        directives.append(
            {
                "scope": directive.get("scope"),
                "order_id": directive.get("order_id"),
                "tags": directive.get("tags") or directive.get("derived_tags") or [],
                "target_refs": directive.get("target_refs") or [],
            }
        )
    state = {
        "activation_id": user_payload.get("activation_id"),
        "mission": user_payload.get("mission") or {},
        "doctrine": user_payload.get("doctrine"),
        "priority_rules": user_payload.get("priority_rules") or [],
        "actor": user_payload.get("actor") or {},
        "round": user_payload.get("round"),
        "directives": directives,
    }
    if user_payload.get("battlefield"):
        state["battlefield"] = user_payload.get("battlefield")
    if user_payload.get("actions_remaining"):
        state["actions_remaining"] = user_payload.get("actions_remaining")
    return state


def _select_jev_tactical_option(
    *,
    user_payload: dict[str, Any],
    activation_id: str,
    offered_option_ids: list[str],
) -> dict[str, Any]:
    """Call TypeSafe Jev for a typed choice among backend-offered option IDs."""
    if not settings.llm_external_enabled or not settings.typesafe_api_key:
        raise ProviderError("Jev disabled or missing TypeSafe API key")
    if not offered_option_ids:
        raise ProviderError("No offered tactical options")

    option_by_id = {
        str(option.get("option_id")): option
        for option in _listed_options(user_payload)
        if option.get("option_id") in offered_option_ids
    }
    criteria = {option_id: _criteria_blob(option) for option_id, option in option_by_id.items()}
    # A missing description must never cause a legal option to disappear from Jev's choice set.
    for option_id in offered_option_ids:
        criteria.setdefault(option_id, "Backend-approved tactical option.")

    body = {
        "model": settings.typesafe_model,
        "state": _jev_state(user_payload),
        "questions": {
            "tactical_option": {
                "type": "choice",
                "instructions": (
                    "Choose the best legal tactical option for this activation. "
                    "The criteria keys are opaque backend option IDs; select exactly one."
                ),
                "criteria": criteria,
            }
        },
    }
    url = settings.typesafe_base_url.rstrip("/") + "/v1/systemone"
    headers = {
        "Authorization": f"Bearer {settings.typesafe_api_key}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=settings.jev_decision_timeout_sec) as client:
            resp = client.post(url, headers=headers, json=body)
            if resp.status_code >= 400:
                raise ProviderError(f"TypeSafe {resp.status_code}: {resp.text[:400]}")
            data = resp.json()
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"TypeSafe request failed: {exc}") from exc

    try:
        answer = data["answers"]["tactical_option"]
        option_id = str(answer["choice"])
        confidence = float(answer["confidence"])
        probabilities = {str(key): float(value) for key, value in answer["probabilities"].items()}
    except Exception as exc:
        raise ProviderError(f"Malformed TypeSafe choice response: {exc}") from exc
    if option_id not in offered_option_ids:
        raise ProviderError(f"Unoffered TypeSafe option_id {option_id}")
    selected_probability = probabilities.get(option_id)
    if selected_probability is None:
        raise ProviderError("TypeSafe response omitted selected-option probability")
    other_probabilities = [value for key, value in probabilities.items() if key != option_id]
    margin = selected_probability - max(other_probabilities, default=0.0)
    if confidence < settings.jev_min_confidence:
        raise ProviderError(f"TypeSafe confidence {confidence:.3f} below threshold")
    if len(offered_option_ids) > 1 and margin < settings.jev_min_margin:
        raise ProviderError(f"TypeSafe choice margin {margin:.3f} below threshold")
    return {
        "option_id": option_id,
        "fallback_policy": "hold",
        "reason": "typed_typesafe_choice",
        "raw": {
            "model": data.get("model", settings.typesafe_model),
            "usage": data.get("usage") or {},
            "provider": "typesafe",
            "confidence": confidence,
            "margin": margin,
            "probabilities": probabilities,
        },
    }


def generate_radio_line(*, facts: dict[str, Any], unit_name: str) -> str:
    if radio_provider_name() != "openai":
        return _fallback_radio(facts, unit_name)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly squad/drone radio voice. "
                "Reply with ONE short factual acknowledgement sentence based only on the provided facts. "
                "No coordinates invention. No dice talk."
            ),
        },
        {"role": "user", "content": json.dumps({"unit": unit_name, "facts": facts})},
    ]
    body = {
        "model": settings.llm_model_radio,
        "messages": messages,
        "reasoning_effort": "none",
        "max_completion_tokens": 60,
    }
    url = settings.openai_base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=settings.llm_radio_timeout_sec) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()
            if not text:
                return _fallback_radio(facts, unit_name)
            return text.split("\n")[0][:180]
    except Exception:
        return _fallback_radio(facts, unit_name)


def _fallback_radio(facts: dict[str, Any], unit_name: str) -> str:
    sub = facts.get("subroutine", "action")
    return f"{unit_name}: {sub} complete."
