"""Dedicated LLM agent for the Red Opposition Commander."""

from __future__ import annotations

from typing import Any

import yaml
from sqlalchemy.orm import Session

from ..config import settings
from ..content.loader import content_root, get_catalog
from ..domain.enums import ActionType, Side
from ..domain.events import DomainEvent
from ..domain.hex import Hex, axial_distance
from ..engine.battle import end_activation, execute_option
from ..engine.options import build_options, fallback_select
from ..engine.state import (
    BattleState,
    UnitState,
    commander_for_side,
    commander_ram_abilities,
    in_signal,
    signal_radius,
    unit_within_radius,
    within_commander_signal,
)
from ..persistence.models import AgentRunRow, LLMArtifactRow
from ..telemetry.logging import Timer, hash_payload, redact_text, structured_log, trace_id_var
from . import provider as llm
from .orchestration import maybe_resolve_opposition_control_phase

# Heavy cannon engagement band — close enough to shoot, not suicide-melee.
CANNON_RANGE = 10
SAFE_STANDOFF = 6
MAX_ACTIONS_PER_ACTIVATION = 3
# Enemy this close to an ally tank/drone/squad counts as "engaged".
ALLY_ENGAGED_RANGE = 12
# Soft north depth only when the commander has no living screen left.
OPPOSITION_SAFE_DEPTH_R = 18


def _unit_max_hp(unit: UnitState) -> int:
    return sum(m.max_hp for m in unit.models) if unit.models else _unit_hp(unit)


def _ally_screen(battle: BattleState, ally: Side) -> list[UnitState]:
    return [
        u
        for u in battle.living_units(ally)
        if u.alive and u.category in ("soldier_squad", "drone")
    ]


def _living_enemies(battle: BattleState, enemy: Side) -> list[UnitState]:
    return [u for u in battle.living_units(enemy) if u.alive and u.category != "decoy"]


def _allies_engaged(battle: BattleState, ally: Side, enemy: Side) -> bool:
    screen = _ally_screen(battle, ally)
    foes = _living_enemies(battle, enemy)
    if not screen or not foes:
        return False
    return any(
        axial_distance(a.position, e.position) <= ALLY_ENGAGED_RANGE for a in screen for e in foes
    )


def _max_screen_distance(commander: UnitState, screen: list[UnitState]) -> int | None:
    if not screen:
        return None
    return max(axial_distance(commander.position, u.position) for u in screen)


def _signal_leash_broken(commander: UnitState, screen: list[UnitState], sig_r: int) -> bool:
    """True when any screen unit is farther than signal radius (cannot allocate RAM)."""
    dist = _max_screen_distance(commander, screen)
    return dist is not None and dist > sig_r


def _too_dangerous_to_advance(battle: BattleState, commander: UnitState, enemy: Side) -> bool:
    hp = _unit_hp(commander)
    max_hp = max(1, _unit_max_hp(commander))
    if hp <= max(1, max_hp // 3):
        return True
    return any(
        axial_distance(commander.position, e.position) <= SAFE_STANDOFF - 1
        for e in _living_enemies(battle, enemy)
    )


def _system_prompt() -> str:
    path = content_root() / "prompts" / "opposition_commander.yaml"
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data.get("system", "Select one commander option via tool call.")
    return "Select one commander option via tool call."


def _enemy_side(commander: UnitState) -> Side:
    return Side.OPPOSITION if commander.side == Side.FRIENDLY else Side.FRIENDLY


def _ally_side(commander: UnitState) -> Side:
    return commander.side


def _unit_hp(unit: UnitState) -> int:
    return sum(m.hp for m in unit.living_models) if unit.living_models else 0


def _enabled_options(options: dict[str, dict], sub: str | None = None) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for oid, opt in options.items():
        if sub is not None and opt.get("subroutine") != sub:
            continue
        prev = opt.get("preview") or {}
        if prev.get("disabled") or prev.get("blocked_reason"):
            continue
        out.append((oid, opt))
    return out


def _embed_ram_targets(battle: BattleState, commander: UnitState, options: dict[str, dict]) -> None:
    """Fill target_unit_id on RAM options so the engine can auto-resolve for the AI."""
    enemy = _enemy_side(commander)
    for opt in options.values():
        if opt.get("subroutine") != "ram_ability":
            continue
        preview = opt.get("preview") or {}
        if preview.get("disabled"):
            continue
        aid = preview.get("ability_id")
        if aid == "airstrike" and preview.get("needs_target"):
            painted = [
                u
                for u in battle.living_units(enemy)
                if "painted" in u.statuses and u.category != "decoy"
            ]
            painted.sort(key=lambda u: (-_unit_hp(u), u.unit_instance_id))
            if painted:
                preview["target_unit_id"] = painted[0].unit_instance_id
                preview["needs_target"] = False
        elif aid == "signal_jamming" and preview.get("needs_target"):
            drones = [
                u
                for u in battle.living_units(enemy)
                if u.category == "drone"
                and u.alive
                and within_commander_signal(battle, u.position, side=commander.side)
            ]
            drones.sort(
                key=lambda u: (
                    axial_distance(commander.position, u.position),
                    u.unit_instance_id,
                )
            )
            if drones:
                preview["target_unit_id"] = drones[0].unit_instance_id
                preview["needs_target"] = False


def _battlefield_snapshot(battle: BattleState, commander: UnitState) -> dict[str, Any]:
    ally = _ally_side(commander)
    enemy = _enemy_side(commander)
    blue_cmd = commander_for_side(battle, Side.FRIENDLY)
    drones = [u for u in battle.living_units(ally) if u.category == "drone"]
    drones_in_sig = [u for u in drones if in_signal(battle, u)]
    soldiers = [u for u in battle.living_units(ally) if u.category == "soldier_squad"]
    sig_r = signal_radius(commander)
    threats = [
        u
        for u in battle.living_units(enemy)
        if u.alive and u.category != "decoy" and axial_distance(commander.position, u.position) <= sig_r + 4
    ]
    in_cannon = [
        u
        for u in battle.living_units(enemy)
        if u.alive and u.category != "decoy" and axial_distance(commander.position, u.position) <= CANNON_RANGE
    ]
    painted_enemies = [
        u for u in battle.living_units(enemy) if "painted" in u.statuses and u.category != "decoy"
    ]
    screen = _ally_screen(battle, ally)
    max_screen = _max_screen_distance(commander, screen)
    return {
        "commander": {
            "id": commander.unit_instance_id,
            "hp": _unit_hp(commander),
            "position": commander.position.to_dict(),
            "ram_current": commander.ram_current,
            "ram_capacity": commander.ram_capacity,
            "signal_radius": sig_r,
            "statuses": list(commander.statuses),
            "weapon": "heavy_cannon",
            "cannon_range": CANNON_RANGE,
        },
        "enemy_commander": (
            {
                "id": blue_cmd.unit_instance_id,
                "hp": _unit_hp(blue_cmd),
                "position": blue_cmd.position.to_dict(),
                "distance": axial_distance(commander.position, blue_cmd.position),
            }
            if blue_cmd and blue_cmd.alive
            else None
        ),
        "enemies_in_cannon_range": [
            {
                "id": u.unit_instance_id,
                "name": u.display_name,
                "distance": axial_distance(commander.position, u.position),
                "category": u.category,
            }
            for u in sorted(in_cannon, key=lambda u: axial_distance(commander.position, u.position))[:8]
        ],
        "drones_in_signal": len(drones_in_sig),
        "drones_total": len(drones),
        "soldiers_in_signal": len(
            [u for u in soldiers if unit_within_radius(commander.position, u, sig_r)]
        ),
        "screen_units": len(screen),
        "max_distance_to_screen": max_screen,
        "signal_leash_broken": _signal_leash_broken(commander, screen, sig_r),
        "allies_engaged": _allies_engaged(battle, ally, enemy),
        "too_dangerous_to_advance": _too_dangerous_to_advance(battle, commander, enemy),
        "allocated_ram_on_drones": [
            {"id": u.unit_instance_id, "name": u.display_name, "allocated_ram": u.allocated_ram}
            for u in drones
            if u.allocated_ram
        ],
        "ram_abilities": commander_ram_abilities(battle, commander),
        "nearby_threats": [
            {
                "id": u.unit_instance_id,
                "name": u.display_name,
                "distance": axial_distance(commander.position, u.position),
                "category": u.category,
            }
            for u in sorted(threats, key=lambda u: axial_distance(commander.position, u.position))[:8]
        ],
        "painted_enemy_targets": [u.unit_instance_id for u in painted_enemies],
        "vp": {
            "friendly": getattr(battle, "friendly_vp", 0),
            "opposition": getattr(battle, "opposition_vp", 0),
            "to_win": getattr(battle, "vp_to_win", 5),
        },
        "objective_center": {"q": battle.width // 2, "r": battle.height // 2},
    }


def _pick_best_attack(attacks: list[tuple[str, dict]], battle: BattleState, commander: UnitState) -> str | None:
    if not attacks:
        return None
    blue_cmd = commander_for_side(battle, Side.FRIENDLY)
    scored: list[tuple[int, str]] = []
    for oid, opt in attacks:
        prev = opt.get("preview") or {}
        tid = prev.get("target_unit_id")
        score = 10
        if blue_cmd and tid == blue_cmd.unit_instance_id:
            score = 10_000
        elif tid and tid in battle.units:
            tgt = battle.units[tid]
            score = 1000 if tgt.category == "commander" else 500
            score += _unit_hp(tgt) * 10
            score -= axial_distance(commander.position, tgt.position)
            if tgt.category == "drone":
                score += 50
        scored.append((score, oid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1]


def _pick_ram(ram_opts: list[tuple[str, dict]], ability_id: str) -> str | None:
    for oid, opt in ram_opts:
        if (opt.get("preview") or {}).get("ability_id") == ability_id:
            return oid
    return None


def _is_dash_move(opt: dict) -> bool:
    prev = opt.get("preview") or {}
    if prev.get("dash"):
        return True
    return int(prev.get("moves_required") or 1) >= 2


def _single_moves(moves: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Conservative commander never burns Attack on a dash."""
    singles = [(oid, opt) for oid, opt in moves if not _is_dash_move(opt)]
    return singles or moves


def _ally_front_r(battle: BattleState, ally: Side) -> int | None:
    """Southernmost living ally row (higher r = farther toward Blue)."""
    screen = [
        u
        for u in battle.living_units(ally)
        if u.alive and u.category in ("soldier_squad", "drone")
    ]
    if not screen:
        return None
    return max(u.position.r for u in screen)


def _pick_hold(options: dict[str, dict]) -> str | None:
    for oid, opt in options.items():
        if opt.get("subroutine") == "hold":
            return oid
    return None


def _score_support_move(
    h: Hex,
    *,
    battle: BattleState,
    commander: UnitState,
    ally: Side,
    enemies: list[UnitState],
    ally_screen: list[UnitState],
    front_r: int | None,
    sig_r: int,
    center: Hex,
    blue_cmd: UnitState | None,
    allies_engaged: bool,
    leash_broken: bool,
) -> int:
    """Score a Move hex: keep signal leash, support engaged allies, stay behind screen."""
    score = 0
    if ally_screen:
        max_dist = max(axial_distance(h, a.position) for a in ally_screen)
        if max_dist > sig_r:
            # Must close the leash — every hex over signal radius hurts hard.
            score -= (max_dist - sig_r) * 90
        else:
            # Prefer sitting 4–10 hexes behind the pack (still inside signal).
            score += max(0, 14 - abs(max_dist - 8)) * 8
        near_ally = min(axial_distance(h, a.position) for a in ally_screen)
        score += max(0, 10 - near_ally) * 12
    elif h.r > OPPOSITION_SAFE_DEPTH_R:
        score -= (h.r - OPPOSITION_SAFE_DEPTH_R) * 25

    if front_r is not None:
        if h.r > front_r + 2:
            # Do not leapfrog the screen into melee.
            score -= (h.r - front_r - 2) * 70
        else:
            behind = front_r - h.r
            if 0 <= behind <= sig_r:
                score += 100
                # Ideal: a few hexes behind the tip of the screen.
                score += max(0, 8 - abs(behind - 6)) * 12
            elif behind > sig_r:
                score -= (behind - sig_r) * 40

    if enemies:
        nearest = min(axial_distance(h, e.position) for e in enemies)
        if nearest <= CANNON_RANGE:
            score += 160 if allies_engaged else 90
            if nearest < SAFE_STANDOFF:
                score -= (SAFE_STANDOFF - nearest) * 55
            else:
                score += max(0, 12 - abs(nearest - 8)) * 6
        elif allies_engaged or leash_broken:
            gap = nearest - CANNON_RANGE
            score += max(0, 40 - gap * 4)
        else:
            gap = nearest - CANNON_RANGE
            score -= min(gap, 10) * 3

    # Soft center penalty — ignore when we must rejoin / support.
    dist_center = axial_distance(h, center)
    if dist_center <= 6 and not allies_engaged and not leash_broken:
        score -= (7 - dist_center) * 25

    score += sum(
        12
        for d in battle.living_units(ally)
        if d.category == "drone" and unit_within_radius(h, d, sig_r)
    )

    if blue_cmd and blue_cmd.alive:
        d_cmd = axial_distance(h, blue_cmd.position)
        if SAFE_STANDOFF <= d_cmd <= CANNON_RANGE:
            score += 80
        elif d_cmd < SAFE_STANDOFF:
            score -= (SAFE_STANDOFF - d_cmd) * 45

    return score


def _pick_best_support_move(
    moves: list[tuple[str, dict]],
    battle: BattleState,
    commander: UnitState,
    *,
    allies_engaged: bool,
    leash_broken: bool,
) -> str | None:
    if not moves:
        return None
    ally = _ally_side(commander)
    enemy = _enemy_side(commander)
    sig_r = signal_radius(commander)
    center = Hex(battle.width // 2, battle.height // 2)
    enemies = _living_enemies(battle, enemy)
    ally_screen = _ally_screen(battle, ally)
    front_r = _ally_front_r(battle, ally)
    blue_cmd = commander_for_side(battle, Side.FRIENDLY)
    best_oid = moves[0][0]
    best_score = -10**9
    for oid, opt in moves:
        dest = (opt.get("preview") or {}).get("affected_hexes") or []
        if not dest:
            continue
        h = Hex(dest[0]["q"], dest[0]["r"])
        score = _score_support_move(
            h,
            battle=battle,
            commander=commander,
            ally=ally,
            enemies=enemies,
            ally_screen=ally_screen,
            front_r=front_r,
            sig_r=sig_r,
            center=center,
            blue_cmd=blue_cmd if blue_cmd and blue_cmd.alive else None,
            allies_engaged=allies_engaged,
            leash_broken=leash_broken,
        )
        if score > best_score:
            best_score = score
            best_oid = oid
    return best_oid


def commander_fallback_select(options: dict[str, dict], battle: BattleState, commander: UnitState) -> str:
    """Signal-leash fire support: stay ≤ signal of tanks, advance to shoot when they are engaged."""
    if not options:
        raise ValueError("No options")

    enemy = _enemy_side(commander)
    ally = _ally_side(commander)
    sig_r = signal_radius(commander)
    hp = _unit_hp(commander)
    acts = battle.activation.actions if battle.activation else None
    already_moved = bool(acts and int(acts.moves_spent or 0) >= 1)

    ram_opts = _enabled_options(options, "ram_ability")
    attacks = _enabled_options(options, "attack")
    moves = _single_moves(_enabled_options(options, "move"))

    threats_close = [
        u
        for u in _living_enemies(battle, enemy)
        if axial_distance(commander.position, u.position) <= 3
    ]
    max_hp = max(1, _unit_max_hp(commander))
    threatened = hp <= max(1, max_hp // 2) or len(threats_close) >= 1
    ally_screen = _ally_screen(battle, ally)
    enemies = _living_enemies(battle, enemy)
    leash_broken = _signal_leash_broken(commander, ally_screen, sig_r)
    allies_engaged = _allies_engaged(battle, ally, enemy)
    in_cannon_now = any(axial_distance(commander.position, e.position) <= CANNON_RANGE for e in enemies)
    dangerous = _too_dangerous_to_advance(battle, commander, enemy)
    must_rejoin = leash_broken
    must_support = (
        (allies_engaged or not ally_screen)
        and not in_cannon_now
        and not dangerous
    )

    best_fire = _pick_best_attack(attacks, battle, commander)
    if best_fire:
        return best_fire

    if threatened:
        oid = _pick_ram(ram_opts, "defense_matrix")
        if oid:
            return oid

    # Already moved this activation and cannot shoot -> buff or hold. Never dash.
    if already_moved:
        drones_in_sig = [u for u in battle.living_units(ally) if u.category == "drone" and in_signal(battle, u)]
        if drones_in_sig:
            oid = _pick_ram(ram_opts, "targeting_assistance")
            if oid:
                return oid
        oid = _pick_ram(ram_opts, "signal_jamming")
        if oid:
            return oid
        hold = _pick_hold(options)
        if hold:
            return hold
        return fallback_select(options, commander, battle)

    # Out of signal of tanks/drones, or allies fighting and we cannot shoot yet → advance.
    if (must_rejoin or must_support) and moves and acts and acts.can_spend(ActionType.MOVE):
        oid = _pick_best_support_move(
            moves,
            battle,
            commander,
            allies_engaged=allies_engaged or not ally_screen,
            leash_broken=leash_broken,
        )
        if oid:
            return oid

    drones_in_sig = [u for u in battle.living_units(ally) if u.category == "drone" and in_signal(battle, u)]
    if drones_in_sig and (in_cannon_now or not allies_engaged):
        oid = _pick_ram(ram_opts, "targeting_assistance")
        if oid:
            return oid

    painted = [u for u in battle.living_units(enemy) if "painted" in u.statuses and u.category != "decoy"]
    if painted:
        oid = _pick_ram(ram_opts, "airstrike")
        if oid:
            return oid

    if moves and acts and acts.can_spend(ActionType.MOVE):
        # Keep improving leash / standoff even when not urgent.
        oid = _pick_best_support_move(
            moves,
            battle,
            commander,
            allies_engaged=allies_engaged,
            leash_broken=leash_broken,
        )
        if oid:
            # Hold only when already well leashed and not needed at the gunline.
            dest = ((options.get(oid) or {}).get("preview") or {}).get("affected_hexes") or []
            if dest and not must_rejoin and not must_support and in_cannon_now:
                hold = _pick_hold(options)
                if hold:
                    return hold
            return oid

    hold = _pick_hold(options)
    if hold:
        return hold
    return fallback_select(options, commander, battle)


def _select_one(
    battle: BattleState,
    unit: UnitState,
    options: dict[str, dict],
    *,
    use_llm: bool,
    activation_id: str,
) -> tuple[str, dict[str, Any]]:
    """Pick one option. Returns (option_id, llm_meta)."""
    ally = _ally_side(unit)
    enemy = _enemy_side(unit)
    sig_r = signal_radius(unit)
    screen = _ally_screen(battle, ally)
    enemies = _living_enemies(battle, enemy)
    leash_broken = _signal_leash_broken(unit, screen, sig_r)
    allies_engaged = _allies_engaged(battle, ally, enemy)
    in_cannon_now = any(axial_distance(unit.position, e.position) <= CANNON_RANGE for e in enemies)
    dangerous = _too_dangerous_to_advance(battle, unit, enemy)
    must_advance = (leash_broken or ((allies_engaged or not screen) and not in_cannon_now)) and not dangerous

    filtered: dict[str, dict] = {}
    for oid, opt in options.items():
        prev = opt.get("preview") or {}
        if prev.get("disabled") or prev.get("blocked_reason"):
            continue
        if opt.get("subroutine") == "move" and _is_dash_move(opt):
            continue
        # Do not let the LLM park on Hold while out of signal or unable to support.
        if must_advance and opt.get("subroutine") == "hold":
            continue
        filtered[oid] = opt
    menu_source = filtered or {oid: opt for oid, opt in _enabled_options(options)}

    enabled_ids = list(menu_source.keys())
    meta: dict[str, Any] = {
        "fallback_used": False,
        "fallback_reason": None,
        "provider": "fallback",
        "model": "deterministic",
        "prompt_payload": None,
        "prompt_hash": "",
        "response_hash": "",
        "token_usage": {},
        "llm_error": None,
        "llm_success": False,
        "llm_duration": 0.0,
    }
    if use_llm and enabled_ids and llm.tactical_provider_ready():
        try:
            offered_menu = [
                {
                    "option_id": oid,
                    "label": redact_text(str(opt.get("label", oid)), 120),
                    "subroutine": opt.get("subroutine"),
                    "preview": {
                        k: opt.get("preview", {}).get(k)
                        for k in (
                            "ability_id",
                            "target_unit_id",
                            "weapon_id",
                            "dash",
                            "closes_on_objective",
                            "enters_objective",
                        )
                        if k in (opt.get("preview") or {})
                    },
                }
                for oid, opt in menu_source.items()
            ][:40]
            priority_rules = [
                "Fire Heavy Cannon whenever offered — prefer Blue commander, then tanks fighting your screen.",
                "Keep tanks/drones within your signal radius (usually 12). If signal_leash_broken, Move closer — never Hold.",
                "When allies_engaged and you cannot Fire yet, advance carefully into cannon range (6–10 hex standoff) if not too_dangerous_to_advance.",
                "Never Double-move / Dash (dash options are not offered). One Move max, then Fire or RAM.",
                "Stay behind your front line — do not leapfrog tanks into melee.",
                "Targeting Assistance is fine when already leashed; do not buff instead of rejoining signal or supporting a fight.",
                "Hold ONLY when already within signal of your screen AND (you can Fire OR allies are not engaged).",
            ]
            if must_advance:
                priority_rules.insert(
                    0,
                    "CRITICAL: You must Move this step — restore signal leash and/or enter cannon range to support engaged allies. Hold is not offered.",
                )
            prompt_payload = {
                "activation_id": activation_id,
                "battlefield": _battlefield_snapshot(battle, unit),
                "actions_remaining": battle.activation.actions.to_dict() if battle.activation else {},
                "round": battle.round,
                "offered_options": offered_menu,
                "priority_rules": priority_rules,
            }
            meta["prompt_payload"] = prompt_payload
            meta["prompt_hash"] = hash_payload(prompt_payload)
            llm_timer = Timer()
            result = llm.select_tactical_option(
                system_prompt=_system_prompt(),
                user_payload=prompt_payload,
                activation_id=activation_id,
                offered_option_ids=enabled_ids,
            )
            meta["llm_duration"] = llm_timer.ms()
            raw = result.get("raw") or {}
            meta["provider"] = raw.get("provider", "openai")
            meta["model"] = raw.get("model", settings.llm_model_tactical)
            meta["token_usage"] = raw.get("usage") or {}
            meta["confidence"] = raw.get("confidence")
            meta["margin"] = raw.get("margin")
            meta["response_hash"] = hash_payload(result)
            meta["llm_success"] = True
            selected = result["option_id"]
            if selected not in enabled_ids:
                raise llm.ProviderError(f"disabled_or_unknown option {selected}")
            if llm.tactical_shadow_enabled():
                committed = commander_fallback_select(options, battle, unit)
                meta["shadow"] = True
                meta["jev_option_id"] = selected
                meta["agreement"] = committed == selected
                meta["provider"] = "shadow"
                return committed, meta
            return selected, meta
        except Exception as exc:
            meta["fallback_used"] = True
            meta["fallback_reason"] = str(exc)[:500]
            meta["llm_error"] = meta["fallback_reason"]
            meta["provider"] = "fallback"
            meta["model"] = "deterministic"
    else:
        meta["fallback_used"] = True
        meta["fallback_reason"] = "llm_disabled_or_no_options"

    return commander_fallback_select(options, battle, unit), meta


def run_opposition_commander_activation(battle: BattleState, db: Session | None = None) -> list[dict]:
    """Resolve Red commander Control Phase + multi-action tactical activation."""
    catalog = get_catalog()
    if not battle.activation:
        return []
    unit = battle.units.get(battle.activation.actor_id)
    if not unit or unit.category != "commander" or unit.side != Side.OPPOSITION:
        return []

    envelopes: list[dict] = []
    cp_events = maybe_resolve_opposition_control_phase(battle)
    if cp_events:
        envelopes.extend(cp_events)

    timer = Timer()
    activation_id = battle.activation.activation_id
    issued_version = battle.state_version
    use_llm = True
    actions_taken: list[str] = []
    last_selected: str | None = None
    last_subroutine: str | None = None
    fired_this_activation = False
    moved_this_activation = False
    fallback_used = False
    fallback_reason: str | None = None
    shadow_observed = False
    confidence = None
    margin = None
    jev_option_id = None
    agreement = None
    provider_name = "fallback"
    model_name = "deterministic"
    prompt_payload: dict | None = None
    prompt_hash = ""
    response_hash = ""
    token_usage: dict = {}
    llm_error = None
    llm_success = False
    llm_duration = 0.0
    offered_count = 0

    structured_log(
        "opposition_commander_activation_started",
        battle_id=battle.battle_id,
        actor_id=unit.unit_instance_id,
        round=battle.round,
    )

    for step in range(MAX_ACTIONS_PER_ACTIVATION):
        if battle.status.value != "ACTIVE" or not battle.activation:
            break
        if battle.activation.actor_id != unit.unit_instance_id or not unit.alive:
            break

        options = build_options(catalog, battle, unit)
        _embed_ram_targets(battle, unit, options)
        battle.activation.options = options
        enabled = _enabled_options(options)
        offered_count = max(offered_count, len(options))
        if not enabled:
            break

        attacks = _enabled_options(options, "attack")
        ram_opts = _enabled_options(options, "ram_ability")

        if moved_this_activation and not fired_this_activation:
            if attacks:
                selected = _pick_best_attack(attacks, battle, unit)
                meta = {"fallback_used": False, "provider": provider_name, "model": model_name}
                assert selected
            else:
                oid = None
                if threatened_need_move(battle, unit):
                    oid = _pick_ram(ram_opts, "defense_matrix")
                if not oid and any(
                    u.category == "drone" and in_signal(battle, u)
                    for u in battle.living_units(unit.side)
                ):
                    oid = _pick_ram(ram_opts, "targeting_assistance")
                if not oid:
                    oid = _pick_hold(options)
                if not oid:
                    break
                selected = oid
                meta = {"fallback_used": False, "provider": provider_name, "model": model_name}
        elif fired_this_activation:
            if threatened_need_move(battle, unit):
                oid = _pick_ram(ram_opts, "defense_matrix")
                if oid:
                    selected = oid
                    meta = {"fallback_used": False, "provider": provider_name, "model": model_name}
                else:
                    break
            else:
                break
        else:
            selected, meta = _select_one(
                battle, unit, options, use_llm=use_llm and step == 0, activation_id=activation_id
            )
            if meta.get("prompt_payload") is not None:
                prompt_payload = meta["prompt_payload"]
                prompt_hash = meta.get("prompt_hash") or prompt_hash
                response_hash = meta.get("response_hash") or response_hash
                token_usage = meta.get("token_usage") or token_usage
                llm_success = bool(meta.get("llm_success"))
                llm_duration = float(meta.get("llm_duration") or 0)
                llm_error = meta.get("llm_error")
            if meta.get("shadow"):
                shadow_observed = True
                confidence = meta.get("confidence")
                margin = meta.get("margin")
                jev_option_id = meta.get("jev_option_id")
                agreement = meta.get("agreement")
                battle.append_events(
                    [
                        DomainEvent(
                            type="agent_shadow_observed",
                            actor_id=unit.unit_instance_id,
                            payload={
                                "jev_option_id": jev_option_id,
                                "committed_option_id": selected,
                                "agreement": agreement,
                                "confidence": confidence,
                                "margin": margin,
                                "provider_latency_ms": meta.get("llm_duration"),
                                "model": meta.get("model"),
                                "agent": "opposition_commander",
                            },
                        )
                    ]
                )
            elif meta.get("llm_success"):
                confidence = meta.get("confidence", confidence)
                margin = meta.get("margin", margin)
                battle.append_events(
                    [
                        DomainEvent(
                            type="agent_choice_received",
                            actor_id=unit.unit_instance_id,
                            payload={
                                "option_id": selected,
                                "provider": meta.get("provider"),
                                "model": meta.get("model"),
                                "agent": "opposition_commander",
                            },
                        )
                    ]
                )
            if meta.get("fallback_used"):
                fallback_used = True
                fallback_reason = (fallback_reason or "") + f";{meta.get('fallback_reason')}"
                if step == 0:
                    battle.append_events(
                        [
                            DomainEvent(
                                type="agent_fallback_used",
                                actor_id=unit.unit_instance_id,
                                payload={
                                    "reason": meta.get("fallback_reason"),
                                    "option_id": selected,
                                    "agent": "opposition_commander",
                                },
                            )
                        ]
                    )
            provider_name = meta.get("provider") or provider_name
            model_name = meta.get("model") or model_name

            if options.get(selected, {}).get("subroutine") == "move" and _is_dash_move(options[selected]):
                selected = commander_fallback_select(options, battle, unit)
                fallback_used = True
                fallback_reason = (fallback_reason or "") + ";dash_veto"

        if selected not in battle.activation.options:
            selected = commander_fallback_select(options, battle, unit)
            fallback_used = True

        subroutine = options.get(selected, {}).get("subroutine")
        if subroutine == "hold":
            try:
                envelopes.extend(execute_option(battle, selected))
            except ValueError:
                pass
            last_selected = selected
            last_subroutine = "hold"
            break

        try:
            envelopes.extend(execute_option(battle, selected))
        except ValueError as exc:
            fallback_used = True
            fallback_reason = (fallback_reason or "") + f";execute_failed:{exc}"
            options = build_options(catalog, battle, unit)
            _embed_ram_targets(battle, unit, options)
            battle.activation.options = options
            attacks = _enabled_options(options, "attack")
            recover = _pick_best_attack(attacks, battle, unit) or _pick_hold(options)
            if recover:
                try:
                    envelopes.extend(execute_option(battle, recover))
                    selected = recover
                    subroutine = options.get(recover, {}).get("subroutine")
                except ValueError:
                    break
            else:
                break

        actions_taken.append(f"{subroutine}:{selected}")
        last_selected = selected
        last_subroutine = subroutine
        if subroutine == "attack":
            fired_this_activation = True
        if subroutine == "move":
            moved_this_activation = True

        if not battle.activation:
            break
        acts = battle.activation.actions
        if not (
            acts.can_spend(ActionType.STANDARD)
            or acts.can_spend(ActionType.MOVE)
            or acts.can_spend(ActionType.MINOR)
        ):
            break

    entry = {
        "speaker": unit.display_name,
        "side": "opposition",
        "text": f"{unit.display_name}: {', '.join(actions_taken) if actions_taken else 'standing by'}.",
        "unit_id": unit.unit_instance_id,
    }
    battle.communications.append(entry)
    battle.append_events([DomainEvent(type="communication_added", payload=entry)])

    if battle.status.value == "ACTIVE" and battle.activation and battle.activation.actor_id == unit.unit_instance_id:
        end_activation(battle)

    duration = timer.ms()
    run_status = "FALLBACK" if fallback_used else ("SHADOW" if shadow_observed else "COMMITTED")
    if db is not None:
        run = AgentRunRow(
            session_id=battle.session_id,
            battle_id=battle.battle_id,
            activation_id=activation_id,
            actor_id=unit.unit_instance_id,
            round=battle.round,
            side=unit.side.value,
            definition_id=unit.definition_id,
            status=run_status,
            selected_option_id=last_selected,
            selected_subroutine=last_subroutine,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            provider=provider_name,
            model=model_name,
            offered_option_count=offered_count,
            issued_state_version=issued_version,
            committed_state_version=battle.state_version,
            duration_ms=duration,
            trace_id=trace_id_var.get(),
            artifact_json={
                "agent": "opposition_commander",
                "actions_taken": actions_taken,
                "confidence": confidence,
                "margin": margin,
                "provider_latency_ms": llm_duration or None,
                "shadow": shadow_observed,
                "jev_option_id": jev_option_id,
                "agreement": agreement,
            },
        )
        db.add(run)
        db.flush()
        if prompt_payload is not None or llm_error:
            db.add(
                LLMArtifactRow(
                    session_id=battle.session_id,
                    battle_id=battle.battle_id,
                    agent_run_id=run.id,
                    purpose="opposition_commander",
                    provider=provider_name or "none",
                    model=model_name or "",
                    prompt_hash=prompt_hash,
                    response_hash=response_hash,
                    token_usage_json=token_usage if isinstance(token_usage, dict) else {},
                    summary_json={
                        "selected_option_id": last_selected,
                        "subroutine": last_subroutine,
                        "actions_taken": actions_taken,
                        "confidence": confidence,
                        "margin": margin,
                        "provider_latency_ms": llm_duration or None,
                        "shadow": shadow_observed,
                        "agreement": agreement,
                        "fallback_reason": redact_text(fallback_reason or "", 120),
                    },
                    success=llm_success,
                    error=llm_error,
                    duration_ms=llm_duration,
                    trace_id=trace_id_var.get(),
                )
            )

    structured_log(
        "opposition_commander_activation_finished",
        battle_id=battle.battle_id,
        actor_id=unit.unit_instance_id,
        fallback_used=fallback_used,
        subroutine=last_subroutine,
        actions_taken=actions_taken,
        duration_ms=duration,
    )
    return envelopes


def threatened_need_move(battle: BattleState, commander: UnitState) -> bool:
    enemy = _enemy_side(commander)
    return any(
        u.alive and u.category != "decoy" and axial_distance(commander.position, u.position) <= 2
        for u in battle.living_units(enemy)
    )
