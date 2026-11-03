"""Agent activation orchestration with Luna + deterministic fallback + durable run logs."""

from __future__ import annotations

import yaml
from sqlalchemy.orm import Session

from ..config import settings
from ..content.loader import content_root, get_catalog
from ..domain.enums import ActionType, Side
from ..domain.events import DomainEvent
from ..engine.battle import end_activation, execute_option, merge_chained_moves
from ..domain.hex import Hex
from ..engine.options import (
    _attack_target_in_objective,
    _closes_on_enemy,
    _is_disposable_bomber,
    _is_striker,
    build_options,
    fallback_select,
    should_chain_dash,
)
from ..engine.state import BattleState
from ..persistence.models import AgentRunRow, LLMArtifactRow
from ..telemetry.logging import Timer, hash_payload, redact_text, structured_log, trace_id_var
from . import provider as llm


def _system_prompt() -> str:
    path = content_root() / "prompts" / "tactical_system.yaml"
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data.get("system", "Select one tactical option via tool call.")
    return "Select one tactical option via tool call."


def maybe_resolve_opposition_control_phase(battle: BattleState) -> list[dict]:
    """
    Side-agnostic Control Phase hook for the Red commander.

    Auto-allocates up to 1 RAM onto in-signal small strike drones (dogs / burst),
    preferring disposable bombers, then completes Control Phase.
    """
    from ..engine.control_phase import (
        allocate_ram,
        complete_control_phase,
        control_phase_blocks_commander_actions,
        eligible_allocation_drones,
        max_ram_for_drone,
    )

    if not control_phase_blocks_commander_actions(battle):
        return []
    if not battle.activation:
        return []
    unit = battle.units.get(battle.activation.actor_id)
    if not unit or unit.category != "commander" or unit.side != Side.OPPOSITION:
        return []

    events: list = []
    eligible = eligible_allocation_drones(battle, unit) if unit.ram_capacity is not None else []
    # Prefer one-ways, then other small strikers; leave a little RAM for abilities.
    ranked = sorted(
        eligible,
        key=lambda d: (
            0 if "disposable" in (d.roles or []) else 1,
            0 if (d.size_class or "").lower() == "small" else 1,
            0 if {"mobile_damage", "area_damage"} & set(d.roles or []) else 1,
            d.unit_instance_id,
        ),
    )
    reserve = 2  # keep some pool for Call for Action / Defense Matrix
    for drone in ranked:
        cap = max_ram_for_drone(drone)
        while (
            drone.allocated_ram < cap
            and int(unit.ram_current or 0) > reserve
        ):
            try:
                events.extend(allocate_ram(battle, drone.unit_instance_id, actor_side=Side.OPPOSITION))
            except ValueError:
                break

    structured_log(
        "control_phase_auto_allocate",
        battle_id=battle.battle_id,
        session_id=battle.session_id,
        actor_id=unit.unit_instance_id,
        eligible_drone_count=len(eligible),
        allocated=[
            {"id": d.unit_instance_id, "ram": d.allocated_ram}
            for d in ranked
            if d.allocated_ram > 0
        ],
        ram_remaining=unit.ram_current,
    )
    events.extend(complete_control_phase(battle, actor_side=Side.OPPOSITION))
    return battle.append_events(events)


def run_agent_activation(battle: BattleState, db: Session | None = None) -> list[dict]:
    """Resolve the current non-commander activation via LLM or fallback."""
    catalog = get_catalog()
    if not battle.activation:
        return []
    unit = battle.units[battle.activation.actor_id]
    if unit.category == "commander" and unit.side == Side.FRIENDLY:
        return []

    if unit.category == "commander" and unit.side == Side.OPPOSITION:
        from .opposition_commander import run_opposition_commander_activation

        return run_opposition_commander_activation(battle, db)

    timer = Timer()
    activation_id = battle.activation.activation_id
    # Always rebuild — never trust persisted menus (stale Fire options after a dash)
    options = build_options(catalog, battle, unit)
    battle.activation.options = options
    offered = list(options.keys())
    selected = None
    fallback_used = False
    fallback_reason = None
    artifact: dict = {}
    provider_name = ""
    model_name = ""
    run_status = "COMMITTED"
    issued_version = battle.state_version
    prompt_payload = None
    prompt_hash = ""
    response_hash = ""
    token_usage: dict = {}
    llm_error = None
    llm_success = False
    llm_duration = 0.0

    structured_log(
        "agent_activation_started",
        battle_id=battle.battle_id,
        session_id=battle.session_id,
        activation_id=activation_id,
        actor_id=unit.unit_instance_id,
        definition_id=unit.definition_id,
        side=unit.side.value,
        round=battle.round,
        option_count=len(options),
        state_version=battle.state_version,
    )

    try:
        if llm.tactical_provider_ready():
            kill_opts = [
                o
                for o in options.values()
                if o.get("subroutine") in ("attack", "self_destruct", "deploy_mine")
            ]
            mine_opts = [o for o in options.values() if o.get("subroutine") == "deploy_mine"]
            is_engineer = "deploy_mine" in (unit.abilities or [])
            is_disposable_bomber = _is_disposable_bomber(unit)
            enter_temple = [
                o
                for o in options.values()
                if o.get("subroutine") in ("move", "return_to_signal", "return_to_resupply")
                and (o.get("preview") or {}).get("enters_objective")
            ]
            closing_temple = [
                o
                for o in options.values()
                if o.get("subroutine") in ("move", "return_to_signal", "return_to_resupply")
                and (o.get("preview") or {}).get("closes_on_objective")
            ]
            rtb_opts = [
                o
                for o in options.values()
                if o.get("subroutine") == "return_to_resupply"
                or (
                    o.get("subroutine") in ("move", "return_to_signal")
                    and (
                        (o.get("preview") or {}).get("enters_resupply")
                        or (o.get("preview") or {}).get("closes_on_resupply")
                    )
                )
            ]
            order_tags: set[str] = set()
            focus_target_id: str | None = None
            if unit.side == Side.FRIENDLY:
                for d in battle.directives or []:
                    if not d.get("active"):
                        continue
                    if d.get("scope") == "global" or d.get("target_unit_id") == unit.unit_instance_id:
                        order_tags.update(d.get("derived_tags") or [])
                        for ref in d.get("target_refs") or []:
                            if isinstance(ref, dict) and ref.get("kind") == "unit":
                                focus_target_id = ref.get("unit_instance_id") or focus_target_id
            combat_ready = bool(
                {"frontline", "mobile_damage", "area_damage", "disposable"} & set(unit.roles)
            )
            from ..engine.objective import unit_contests_objective
            from ..engine.resupply import must_return_to_resupply, should_return_to_resupply

            need_temple = not any(unit_contests_objective(battle, u) for u in battle.living_units(unit.side))
            rearming = should_return_to_resupply(catalog, unit)
            combat_dry = must_return_to_resupply(catalog, unit)
            zone_kills = [
                o
                for o in kill_opts
                if o.get("subroutine") == "self_destruct" or _attack_target_in_objective(battle, o)
            ]
            gun_opts = [o for o in kill_opts if o.get("subroutine") in ("attack", "self_destruct")]
            moves_spent = int(battle.activation.actions.moves_spent or 0)
            already_moved = moves_spent >= 1
            pushing_temple = (
                need_temple
                and (enter_temple or closing_temple)
                and not (_is_striker(unit) and kill_opts)
                and not is_disposable_bomber
            )
            force_attack_menu = bool(kill_opts) and not pushing_temple and (
                unit.side == Side.OPPOSITION
                or _is_striker(unit)
                or bool(order_tags & {"attack", "engage", "focus_fire"})
                or (combat_ready and not (order_tags & {"hold", "defensive"}))
            ) and not ("paint_target" in order_tags and "paint_target" in (unit.abilities or []))
            paint_opts = [o for o in options.values() if o.get("subroutine") == "paint_target"]
            paint_order = "paint_target" in order_tags and "paint_target" in (unit.abilities or [])
            # Only force RTB when truly combat-dry (no rifle/missile left). Empty
            # anti-armor missiles must not suppress Heavy Rifle Fire or burn RAM on Dash home.
            if combat_dry and rtb_opts and not gun_opts:
                offered_menu = rtb_opts
            elif rearming and rtb_opts and not gun_opts:
                offered_menu = rtb_opts
            elif already_moved and (gun_opts or (is_engineer and mine_opts)):
                # Spent Move once — finish with under-enemy mine or guns. No second dash into the temple.
                under = [o for o in mine_opts if (o.get("preview") or {}).get("under_enemy")]
                if under:
                    offered_menu = under + gun_opts
                elif gun_opts:
                    offered_menu = gun_opts
                else:
                    preemptive = [
                        o
                        for o in mine_opts
                        if not (o.get("preview") or {}).get("in_own_deploy")
                        and (
                            (o.get("preview") or {}).get("near_objective")
                            or (o.get("preview") or {}).get("on_approach")
                        )
                    ][:4]
                    offered_menu = preemptive or list(options.values())
            elif is_engineer and mine_opts:
                push_orders = bool(
                    order_tags & {"advance", "objective", "center", "engage", "attack", "focus_fire"}
                ) or unit.side == Side.OPPOSITION
                under = [o for o in mine_opts if (o.get("preview") or {}).get("under_enemy")]
                # Preemptive only on approach corridors or to help hold the objective — never random / deploy-belt.
                preemptive = [
                    o
                    for o in mine_opts
                    if not (o.get("preview") or {}).get("under_enemy")
                    and not (o.get("preview") or {}).get("in_own_deploy")
                    and (
                        (o.get("preview") or {}).get("near_objective")
                        or (o.get("preview") or {}).get("on_approach")
                    )
                ][:4]
                advance_moves = [
                    o
                    for o in options.values()
                    if o.get("subroutine") in ("move", "return_to_signal")
                    and (
                        (o.get("preview") or {}).get("closes_on_objective")
                        or (o.get("preview") or {}).get("enters_objective")
                    )
                ]
                if under:
                    # Best sapper play: plant on the foe (guns secondary).
                    offered_menu = under + gun_opts
                elif push_orders and advance_moves and gun_opts:
                    # Advance+engage: move up OR shoot if already in range — never empty-mine spam.
                    offered_menu = advance_moves + gun_opts
                elif push_orders and advance_moves:
                    offered_menu = advance_moves + preemptive + gun_opts
                elif need_temple and (enter_temple or closing_temple):
                    offered_menu = (enter_temple or closing_temple) + gun_opts + preemptive
                else:
                    offered_menu = gun_opts + preemptive + (advance_moves if advance_moves else [])
            elif paint_order:
                if focus_target_id:
                    match = [
                        o
                        for o in paint_opts
                        if (o.get("preview") or {}).get("target_unit_id") == focus_target_id
                    ]
                    if match:
                        offered_menu = match
                    else:
                        goal = battle.units.get(focus_target_id)
                        approach = [
                            o
                            for o in options.values()
                            if o.get("subroutine") in ("move", "return_to_signal")
                            and goal
                            and goal.alive
                            and _closes_on_enemy(
                                unit.position,
                                Hex(**o["preview"]["affected_hexes"][0]),
                                [goal],
                            )
                        ]
                        offered_menu = approach or paint_opts or list(options.values())
                elif paint_opts:
                    offered_menu = paint_opts
                else:
                    offered_menu = list(options.values())
            elif is_disposable_bomber:
                suicides = [o for o in kill_opts if o.get("subroutine") == "self_destruct"]
                det_moves = [
                    o
                    for o in options.values()
                    if o.get("subroutine") in ("move", "return_to_signal")
                    and (o.get("preview") or {}).get("detonation_would_hit")
                ]
                hunt_moves = [
                    o
                    for o in options.values()
                    if o.get("subroutine") in ("move", "return_to_signal")
                    and (o.get("preview") or {}).get("closes_on_enemy")
                ]
                if suicides:
                    offered_menu = suicides
                elif det_moves:
                    # Never offer a non-blast dash when a blast hex is reachable
                    offered_menu = det_moves
                elif hunt_moves:
                    offered_menu = hunt_moves
                elif kill_opts:
                    offered_menu = kill_opts
                else:
                    offered_menu = list(options.values())
            elif _is_striker(unit) and kill_opts:
                offered_menu = kill_opts
            elif need_temple and enter_temple and not (force_attack_menu and gun_opts and already_moved):
                offered_menu = enter_temple + zone_kills
            elif need_temple and closing_temple and not (gun_opts and order_tags & {"engage", "attack", "focus_fire"}):
                offered_menu = closing_temple + zone_kills + (gun_opts if order_tags & {"engage", "attack"} else [])
            elif force_attack_menu:
                offered_menu = kill_opts
            else:
                offered_menu = list(options.values())
            offered_menu = [o for o in offered_menu if not (o.get("preview") or {}).get("disabled")]
            ram_bonus = max(0, int(battle.activation.actions.standard or 0) - 1) if unit.category == "drone" else 0
            priority_rules = [
                "Action economy: Move+Attack OR Double-move. Never Move+Move+Attack. A second Move spends Attack.",
                "If both Move and Fire are offered after you already moved once, prefer Fire unless you must reach the objective / rearm.",
            ]
            if ram_bonus > 0:
                priority_rules.insert(
                    0,
                    f"RAM Control Phase granted you {ram_bonus} bonus Standard action(s) this activation — spend ALL Standard actions before the turn ends (e.g. Attack then Attack, or Move then Attack).",
                )
            if rearming and gun_opts:
                priority_rules.insert(
                    0,
                    "A limited-ammo weapon is empty, but you still have Attack options (e.g. Heavy Rifle). KEEP FIRING — do not Return to Resupply or Dash home while Fire is offered. Only rearm when no Attack options remain.",
                )
            if ram_bonus > 0 and gun_opts:
                priority_rules.insert(
                    0,
                    "You still have bonus Standard(s) and Fire is offered — NEVER spend leftover RAM Standards on Move or Return to Resupply. Fire again.",
                )
            if ram_bonus > 0 and is_disposable_bomber:
                priority_rules.insert(
                    0,
                    "RAM boost: you can Triple-move this activation (Move + Move + RAM-Move) then Self-destruct (Minor). Prefer Sprint/Dash onto a detonation_would_hit hex, then Self-destruct. Never leave a bonus Standard unspent while an enemy is still out of blast range.",
                )
            if "paint_target" in order_tags and "paint_target" in (unit.abilities or []):
                priority_rules.insert(
                    0,
                    "Army order: PAINT TARGET for Airstrike. Use Paint Target (Minor) on the designated opposition unit when in range and LOS. Move closer first if needed — painting beats shooting this turn.",
                )
            priority_rules.extend([
                "If you are a Blue Direct Attack Drone with your one shot spent and no Fire options left, Return to deploy to rearm. Other drones do not reload at the deployment belt.",
                "If your side has nobody contesting the objective, take a Move that enters or closes on it — scoring wins games.",
                "Do not shoot from outside while your side is absent from the objective, unless the target is already inside the zone.",
                "If attack options exist and you already contest the objective (or cannot get closer this move), select an attack.",
                "If Self-destruct is listed, an enemy is in the blast — but check friendlies_in_blast. Prefer Self-destruct when friendlies_in_blast is empty or the trade clearly favors you (e.g. killing the enemy commander). Avoid detonating into clustered friendlies.",
                "Prefer the enemy commander as the attack target when listed.",
                "Never select Hold when an attack, self-destruct (without heavy friendly fire), or objective-claiming move is listed.",
            ])
            if is_disposable_bomber:
                bomber_rules = [
                    "You are a one-way attack drone — detonate on clustered enemy troops (maximize enemy_models_in_blast).",
                    "Action economy: Self-destruct costs Minor. You MAY Double-move / Dash (both Move spends) and STILL Self-destruct the same activation.",
                    "Priority 1: Self-destruct when listed — most enemy_models_in_blast, avoid friendly_fire.",
                    "Priority 2: Dash/Move/Sprint onto a hex where detonation_would_hit is true (reach blast range this activation).",
                    "Priority 3: Double-move / closes_on_enemy toward living opposition — NEVER idle with unused Move. Do NOT fly to empty map center.",
                    "Never end activation after a single Move while a second Move or Self-destruct remains.",
                ]
                if ram_bonus > 0:
                    bomber_rules.insert(
                        1,
                        f"You have {ram_bonus} RAM bonus Standard(s): use them as extra Move(s) for a Triple-move approach if the blast hex is still out of double-move range, then Self-destruct.",
                    )
                priority_rules = bomber_rules + priority_rules
            if is_engineer:
                priority_rules = [
                    "You are a Combat Engineer sapper.",
                    "Priority 1: Deploy Mine under_enemy when listed — plant on the opposition, then it detonates.",
                    "Priority 2: If Fire/attack is listed and you already moved once this activation, take the shot. Never double-move when Fire is available.",
                    "Priority 3: With advance/engage orders, Move closer then shoot — or Move to get adjacent for an under_enemy mine next.",
                    "Priority 4: Preemptive Deploy Mine only when near_objective or on_approach (enemy will pass that hex / hold the temple). Never plant in in_own_deploy. Never shuffle with pointless Moves.",
                    "Shotgun Fire beats empty mine plants and confused lateral Moves.",
                ] + priority_rules
            prompt_payload = {
                "activation_id": battle.activation.activation_id,
                "mission": {
                    "objective": "temple_control",
                    "priority": (
                        "One-way bomber: Move/Dash/Sprint (use RAM bonus Standards as extra Moves) onto enemy clusters then Self-destruct (Minor) — ignore empty map center."
                        if is_disposable_bomber
                        else (
                            "Combat Engineer: under-enemy mines first, then move+shoot with guns, preemptive mines only on approach or to hold the objective."
                            if is_engineer
                            else "Control the temple (map center, 5-hex radius). Uncontested control scores 1 VP at round end; 5 VP wins. Destroy the enemy commander or army also wins. If a Fire/attack option is offered and you already contest the temple, take the shot."
                        )
                    ),
                    "center_hex": {"q": battle.width // 2, "r": battle.height // 2},
                    "control_radius": 5,
                    "vp_to_win": 5,
                    "friendly_vp": getattr(battle, "friendly_vp", 0),
                    "opposition_vp": getattr(battle, "opposition_vp", 0),
                },
                "doctrine": (
                    "disposable_bomber_hunt"
                    if is_disposable_bomber
                    else (
                        "engineer_mine_then_guns"
                        if is_engineer
                        else ("claim_temple" if pushing_temple else ("aggressive_shoot_first" if force_attack_menu else "balanced"))
                    )
                ),
                "priority_rules": priority_rules,
                "actor": {
                    "id": unit.unit_instance_id,
                    "name": unit.display_name,
                    "side": unit.side.value,
                    "definition_id": unit.definition_id,
                    "roles": unit.roles,
                    "position": unit.position.to_dict(),
                    "actions_remaining": battle.activation.actions.to_dict(),
                    "ram_bonus_standards": ram_bonus,
                    "moves_spent_this_activation": battle.activation.actions.moves_spent,
                    "can_still_attack": battle.activation.actions.can_spend(ActionType.STANDARD),
                    "can_still_move": battle.activation.actions.can_spend(ActionType.MOVE),
                },
                "round": battle.round,
                "directives": (
                    []
                    if unit.side == Side.OPPOSITION
                    else [
                        {
                            "scope": d.get("scope"),
                            "order_id": d.get("order_id"),
                            "raw_text": redact_text(str(d.get("raw_text", "")), 120),
                            "tags": d.get("derived_tags") or [],
                            "target_refs": d.get("target_refs") or [],
                        }
                        for d in (battle.directives or [])
                        if d.get("active")
                        and (
                            d.get("scope") == "global"
                            or d.get("target_unit_id") == unit.unit_instance_id
                        )
                    ]
                ),
                "legal_options": [
                    {
                        "option_id": o["option_id"],
                        "subroutine": o["subroutine"],
                        "label": o["label"],
                        "action_cost": o["action_cost"],
                        **(
                            {
                                "blast_radius": (o.get("preview") or {}).get("blast_radius"),
                                "enemies_in_blast": (o.get("preview") or {}).get("enemies_in_blast"),
                                "friendlies_in_blast": (o.get("preview") or {}).get("friendlies_in_blast"),
                                "enemy_models_in_blast": (o.get("preview") or {}).get("enemy_models_in_blast"),
                                "friendly_models_in_blast": (o.get("preview") or {}).get("friendly_models_in_blast"),
                                "friendly_fire": (o.get("preview") or {}).get("friendly_fire"),
                                "risk_tags": (o.get("preview") or {}).get("risk_tags") or [],
                            }
                            if o.get("subroutine") == "self_destruct"
                            else (
                                {
                                    "under_enemy": bool((o.get("preview") or {}).get("under_enemy")),
                                    "in_own_deploy": bool((o.get("preview") or {}).get("in_own_deploy")),
                                    "near_objective": bool((o.get("preview") or {}).get("near_objective")),
                                    "on_approach": bool((o.get("preview") or {}).get("on_approach")),
                                    "center_dist": (o.get("preview") or {}).get("center_dist"),
                                    "affected_hexes": (o.get("preview") or {}).get("affected_hexes") or [],
                                    "risk_tags": (o.get("preview") or {}).get("risk_tags") or [],
                                }
                                if o.get("subroutine") == "deploy_mine"
                                else {}
                            )
                        ),
                    }
                    for o in offered_menu
                ],
            }
            offered_for_tool = [o["option_id"] for o in offered_menu]
            prompt_hash = hash_payload({"system": _system_prompt(), "user": prompt_payload})
            llm_timer = Timer()
            result = llm.select_tactical_option(
                system_prompt=_system_prompt(),
                user_payload=prompt_payload,
                activation_id=battle.activation.activation_id,
                offered_option_ids=offered_for_tool,
            )
            llm_duration = llm_timer.ms()
            artifact = dict(result.get("raw") or {})
            artifact["provider_latency_ms"] = llm_duration
            token_usage = artifact.get("usage") or {}
            provider_name = artifact.get("provider", "openai")
            model_name = artifact.get("model", settings.llm_model_tactical)
            llm_success = True
            if llm.tactical_shadow_enabled():
                jev_option_id = result["option_id"]
                selected = fallback_select(options, unit, battle)
                artifact.update(
                    {
                        "shadow": True,
                        "jev_option_id": jev_option_id,
                        "agreement": selected == jev_option_id,
                    }
                )
                provider_name = "shadow"
                run_status = "SHADOW"
                response_hash = hash_payload(
                    {"jev_option_id": jev_option_id, "committed_option_id": selected}
                )
                battle.append_events(
                    [
                        DomainEvent(
                            type="agent_shadow_observed",
                            actor_id=unit.unit_instance_id,
                            payload={
                                "jev_option_id": jev_option_id,
                                "committed_option_id": selected,
                                "agreement": selected == jev_option_id,
                                "confidence": artifact.get("confidence"),
                                "margin": artifact.get("margin"),
                                "provider_latency_ms": llm_duration,
                                "model": model_name,
                                "trace_id": trace_id_var.get(),
                            },
                        )
                    ]
                )
            else:
                selected = result["option_id"]
                response_hash = hash_payload(
                    {"option_id": selected, "fallback_policy": result.get("fallback_policy")}
                )
                battle.append_events(
                    [
                        DomainEvent(
                            type="agent_choice_received",
                            actor_id=unit.unit_instance_id,
                            payload={
                                "option_id": selected,
                                "provider": provider_name,
                                "model": model_name,
                                "trace_id": trace_id_var.get(),
                            },
                        )
                    ]
                )
        else:
            raise llm.ProviderError("disabled")
    except Exception as exc:
        fallback_used = True
        fallback_reason = str(exc)[:500]
        llm_error = fallback_reason
        selected = fallback_select(options, unit, battle)
        provider_name = provider_name or "fallback"
        model_name = model_name or "deterministic"
        run_status = "FALLBACK"
        battle.append_events(
            [
                DomainEvent(
                    type="agent_fallback_used",
                    actor_id=unit.unit_instance_id,
                    payload={
                        "reason": fallback_reason,
                        "option_id": selected,
                        "trace_id": trace_id_var.get(),
                    },
                )
            ]
        )

    if selected not in battle.activation.options:
        options = build_options(catalog, battle, unit)
        battle.activation.options = options
        selected = fallback_select(options, unit, battle)
        fallback_used = True
        fallback_reason = (fallback_reason or "") + "; stale_option_rebuilt"
        run_status = "FALLBACK"
        battle.append_events(
            [
                DomainEvent(
                    type="agent_fallback_used",
                    actor_id=unit.unit_instance_id,
                    payload={
                        "reason": fallback_reason,
                        "option_id": selected,
                        "trace_id": trace_id_var.get(),
                    },
                )
            ]
        )

    subroutine = options.get(selected, {}).get("subroutine")
    try:
        envelopes = execute_option(battle, selected)
    except ValueError as exc:
        # Stale/illegal attack (e.g. target wiped) — try another attack, else hold, so the turn doesn't soft-fail
        fallback_used = True
        fallback_reason = (fallback_reason or "") + f"; execute_failed:{exc}"
        run_status = "FALLBACK"
        options = build_options(catalog, battle, unit)
        battle.activation.options = options
        envelopes = []
        recovered = False
        if subroutine == "attack":
            for oid, opt in options.items():
                if opt.get("subroutine") != "attack":
                    continue
                try:
                    envelopes = execute_option(battle, oid)
                    selected = oid
                    subroutine = "attack"
                    recovered = True
                    break
                except ValueError:
                    continue
        if not recovered:
            hold_id = next((oid for oid, opt in options.items() if opt.get("subroutine") == "hold"), None)
            if hold_id is None:
                options = build_options(catalog, battle, unit)
                battle.activation.options = options
                hold_id = next(oid for oid, opt in options.items() if opt.get("subroutine") == "hold")
            envelopes = execute_option(battle, hold_id)
            selected = hold_id
            subroutine = "hold"

    radio_text = None
    continue_activation = False
    chained_dash = 0

    # Dash: spend the leftover move now (no second LLM call, one combined path).
    # Leave the activation open only to shoot after moving, or to fly home after shooting.
    if (
        battle.status.value == "ACTIVE"
        and battle.activation
        and battle.activation.actor_id == unit.unit_instance_id
        and subroutine != "hold"
    ):
        battle.activation.options = build_options(catalog, battle, unit)
        # RAM-boosted bombers may need several chained Move spends (triple Move).
        max_chain = 4 if _is_disposable_bomber(unit) else 2
        while (
            chained_dash < max_chain
            and should_chain_dash(catalog, battle, unit, subroutine)
        ):
            remaining = battle.activation.options or {}
            try:
                nxt = fallback_select(remaining, unit, battle)
            except ValueError:
                break
            sub2 = remaining.get(nxt, {}).get("subroutine")
            if sub2 not in ("move", "return_to_signal", "return_to_resupply"):
                break
            try:
                extra = execute_option(battle, nxt)
                envelopes.extend(extra)
                selected = nxt
                subroutine = sub2
                chained_dash += 1
            except ValueError:
                break
        merge_chained_moves(envelopes, unit.unit_instance_id)

        # Disposable bombers: if Self-destruct is legal after the approach move, detonate now
        # (do not wait for another resolve-next — red dog was parking adjacent and ending turn).
        if (
            battle.status.value == "ACTIVE"
            and battle.activation
            and battle.activation.actor_id == unit.unit_instance_id
            and _is_disposable_bomber(unit)
            and subroutine in ("move", "return_to_signal")
        ):
            battle.activation.options = build_options(catalog, battle, unit)
            remaining = battle.activation.options or {}
            suicides = [
                (oid, opt)
                for oid, opt in remaining.items()
                if opt.get("subroutine") == "self_destruct"
                and battle.activation.actions.can_spend(ActionType.MINOR)
            ]
            if suicides:
                # Prefer the blast that hits the most enemies / least friendlies
                def _suicide_key(item: tuple[str, dict]) -> tuple:
                    prev = item[1].get("preview") or {}
                    return (
                        -int(prev.get("enemy_models_in_blast") or 0),
                        int(prev.get("friendly_models_in_blast") or 0),
                    )

                suicides.sort(key=_suicide_key)
                oid, _opt = suicides[0]
                try:
                    extra = execute_option(battle, oid)
                    envelopes.extend(extra)
                    selected = oid
                    subroutine = "self_destruct"
                except ValueError:
                    pass

        remaining = battle.activation.options or {} if battle.activation else {}
        has_kill = any(
            o.get("subroutine") in ("attack", "self_destruct", "deploy_mine") for o in remaining.values()
        )
        has_move = any(
            o.get("subroutine") in ("move", "return_to_signal", "return_to_resupply") for o in remaining.values()
        )
        from ..engine.resupply import must_return_to_resupply, should_return_to_resupply

        rearming = should_return_to_resupply(catalog, unit)
        combat_dry = must_return_to_resupply(catalog, unit)
        can_attack = bool(battle.activation and battle.activation.actions.can_spend(ActionType.STANDARD))
        can_move = bool(battle.activation and battle.activation.actions.can_spend(ActionType.MOVE))
        can_minor = bool(battle.activation and battle.activation.actions.can_spend(ActionType.MINOR))

        def _can_execute_kill(opt: dict) -> bool:
            sub = opt.get("subroutine")
            if sub in ("attack", "deploy_mine"):
                return can_attack
            if sub == "self_destruct":
                return can_minor
            return False

        can_kill_now = any(_can_execute_kill(o) for o in remaining.values() if o.get("subroutine") in ("attack", "self_destruct", "deploy_mine"))
        # Keep shooting while Fire is legal — empty missiles alone must not end the activation
        # or force a leftover-RAM dash home while Heavy Rifle (or other guns) can still fire.
        if subroutine in ("attack", "self_destruct", "ram_ability", "deploy_mine"):
            if subroutine == "self_destruct":
                continue_activation = False
            elif can_kill_now:
                continue_activation = True
            elif subroutine == "attack" and combat_dry and can_move and has_move:
                continue_activation = True
            else:
                continue_activation = False
        elif can_kill_now:
            continue_activation = True
        elif combat_dry and rearming and can_move and has_move and subroutine in ("move", "return_to_signal", "return_to_resupply"):
            continue_activation = True
        else:
            continue_activation = False

    if continue_activation:
        # Leave activation open for another resolve-next step (UI can animate each action)
        pass
    else:
        if unit.side == Side.FRIENDLY:
            facts = {"subroutine": subroutine or "action", "fallback": fallback_used}
            try:
                radio_timer = Timer()
                radio_provider = llm.radio_provider_name()
                radio_text = llm.generate_radio_line(facts=facts, unit_name=unit.display_name)
                if db is not None:
                    db.add(
                        LLMArtifactRow(
                            session_id=battle.session_id,
                            battle_id=battle.battle_id,
                            purpose="radio",
                            provider=radio_provider,
                            model=settings.llm_model_radio if radio_provider == "openai" else "template",
                            prompt_hash=hash_payload(facts),
                            response_hash=hash_payload(radio_text),
                            summary_json={"text_preview": redact_text(radio_text, 80), "duration_ms": radio_timer.ms()},
                            success=True,
                            duration_ms=radio_timer.ms(),
                            trace_id=trace_id_var.get(),
                        )
                    )
            except Exception as radio_exc:
                radio_text = f"{unit.display_name}: orders executed."
                structured_log("radio_failed", error=str(radio_exc)[:200], actor_id=unit.unit_instance_id)
            entry = {
                "speaker": unit.display_name,
                "side": "friendly",
                "text": radio_text,
                "unit_id": unit.unit_instance_id,
            }
            battle.communications.append(entry)
            battle.append_events([DomainEvent(type="communication_added", actor_id=unit.unit_instance_id, payload=entry)])
        else:
            entry = {
                "speaker": "SYSTEM",
                "side": "system",
                "text": f"Opposition contact {unit.display_name} acted.",
                "unit_id": unit.unit_instance_id,
            }
            battle.communications.append(entry)
            battle.append_events([DomainEvent(type="communication_added", payload=entry)])
        if battle.status.value == "ACTIVE" and battle.activation and battle.activation.actor_id == unit.unit_instance_id:
            end_activation(battle)

    duration = timer.ms()
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
            selected_option_id=selected,
            selected_subroutine=subroutine,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            provider=provider_name,
            model=model_name,
            offered_option_count=len(offered),
            issued_state_version=issued_version,
            committed_state_version=battle.state_version,
            duration_ms=duration,
            trace_id=trace_id_var.get(),
            artifact_json={
                "option_label": options.get(selected, {}).get("label"),
                "usage": token_usage,
                "radio_preview": redact_text(radio_text or "", 60),
                "chained_dash": chained_dash,
                "confidence": artifact.get("confidence"),
                "margin": artifact.get("margin"),
                "provider_latency_ms": artifact.get("provider_latency_ms"),
                "shadow": bool(artifact.get("shadow")),
                "jev_option_id": artifact.get("jev_option_id"),
                "agreement": artifact.get("agreement"),
            },
        )
        db.add(run)
        db.flush()
        if prompt_payload is not None or llm_error:
            summary = {
                "option_ids_offered": offered[:20],
                "selected_option_id": selected,
                "selected_subroutine": subroutine,
                "reason_preview": redact_text(fallback_reason or "", 120),
                "confidence": artifact.get("confidence"),
                "margin": artifact.get("margin"),
                "provider_latency_ms": artifact.get("provider_latency_ms"),
                "shadow": bool(artifact.get("shadow")),
                "agreement": artifact.get("agreement"),
            }
            if settings.artifact_retention_mode == "full_diagnostic" and prompt_payload is not None:
                summary["prompt_preview"] = prompt_payload
            db.add(
                LLMArtifactRow(
                    session_id=battle.session_id,
                    battle_id=battle.battle_id,
                    agent_run_id=run.id,
                    purpose="tactical",
                    provider=provider_name or "none",
                    model=model_name or "",
                    prompt_hash=prompt_hash,
                    response_hash=response_hash,
                    token_usage_json=token_usage if isinstance(token_usage, dict) else {},
                    summary_json=summary,
                    success=llm_success,
                    error=llm_error,
                    duration_ms=llm_duration,
                    trace_id=trace_id_var.get(),
                )
            )

    structured_log(
        "agent_activation_finished",
        battle_id=battle.battle_id,
        session_id=battle.session_id,
        actor_id=unit.unit_instance_id,
        status=run_status,
        fallback_used=fallback_used,
        subroutine=subroutine,
        duration_ms=duration,
        state_version=battle.state_version,
        provider=provider_name,
        model=model_name,
    )
    return envelopes
