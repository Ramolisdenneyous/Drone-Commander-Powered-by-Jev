import { useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../../api/client";
import { playSfx, playUnitSelectAck, playArmyOrderVo, fireSfxForAttack } from "../../audio/sfx";
import { startBattleMusic, stopBattleMusic } from "../../audio/music";
import { createBattlefield, BattlefieldHost, unitPortraitUrl } from "../../pixi/battlefield";
import { TabletFrame } from "../../components/TabletFrame";
import { useIsPhone } from "../../hooks/useIsPhone";

type Props = {
  battle: any;
  boot?: any;
  onUpdate: (battle: any) => void;
  onError: (msg: string) => void;
};

type Mode = "move" | "attack" | "ram" | "end";

type ActionFlags = {
  move: number;
  standard: number;
  minor: number;
  movesSpent: number;
  canMove: boolean;
  canAttack: boolean;
  canRam: boolean;
  /** Dedicated Move pool empty (turns Move tab red). */
  moveSpentUi: boolean;
  /** Attack no longer legal (turns Attack tab red). */
  attackSpentUi: boolean;
  /** No action left that can fund RAM (turns RAM tab red). */
  ramSpentUi: boolean;
};

function readActionFlags(actions: any): ActionFlags {
  const move = Number(actions?.move || 0);
  const standard = Number(actions?.standard || 0);
  const minor = Number(actions?.minor || 0);
  const movesSpent = Number(actions?.moves_spent || 0);
  const canMove = movesSpent < 2 && (move > 0 || standard > 0);
  const canAttack = standard > 0 && movesSpent < 2;
  const canRam = minor > 0 || move > 0 || standard > 0;
  return {
    move,
    standard,
    minor,
    movesSpent,
    canMove,
    canAttack,
    canRam,
    moveSpentUi: move < 1,
    attackSpentUi: !canAttack,
    ramSpentUi: !canRam,
  };
}

function ramAbilityShortName(opt: any, boot: any): string {
  const aid = opt?.preview?.ability_id;
  const fromBoot = boot?.abilities?.find((a: any) => a.id === aid);
  if (fromBoot?.display_name) return fromBoot.display_name;
  const label = String(opt?.label || "");
  return label.split(" — ")[0].split(" (")[0].trim() || label;
}

/** Progressive tab after spending: Move → Attack → RAM → End Activation. */
function nextCommanderMode(actions: any): Mode {
  const f = readActionFlags(actions);
  if (f.move > 0 && f.canMove) return "move";
  if (f.canAttack) return "attack";
  if (f.canRam) return "ram";
  return "end";
}

function isFriendlyCommander(u: any | undefined) {
  return u?.category === "commander" && u?.side === "friendly";
}

function combatKindFromEvents(events: any[]): "gunfire" | "explosion_small" | "explosion_large" {
  const weapon =
    events.find((e) => e.type === "weapon_fired")?.payload?.weapon_id ||
    events.find((e) => e.payload?.weapon_id)?.payload?.weapon_id ||
    "";
  const aoe = Number(events.find((e) => e.payload?.aoe)?.payload?.aoe || 0);
  const reason = String(events.find((e) => e.type === "unit_defeated")?.payload?.reason || "");
  // Dramatic mine blast SFX + AOE blast sprite for suicides, mine detonations, and AOE munitions
  if (
    weapon === "airstrike" ||
    weapon === "anti_personnel_mine" ||
    weapon === "micro_explosive" ||
    weapon === "squad_grenade" ||
    String(weapon).startsWith("self_destruct") ||
    reason === "self_destruct" ||
    aoe >= 1
  ) {
    return "explosion_large";
  }
  return "gunfire";
}

function needsAgentResolve(b: any) {
  if (!b || b.status !== "ACTIVE" || !b.active_actor_id) return false;
  const units = [...(b.friendly_units || []), ...(b.opposition_units || [])];
  const active = units.find((u: any) => u.unit_instance_id === b.active_actor_id);
  return !isFriendlyCommander(active);
}

export function BattleScreen({ battle, boot, onUpdate, onError }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const pixiRef = useRef<BattlefieldHost | null>(null);
  const battleRef = useRef(battle);
  const busyRef = useRef(false);
  const resolvingRef = useRef(false);
  const followActiveRef = useRef(true);
  const [busy, setBusy] = useState(false);
  const [resolving, setResolving] = useState(false);
  const [resolveCue, setResolveCue] = useState<string | null>(null);
  const [introScanPlaying, setIntroScanPlaying] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [followActive, setFollowActive] = useState(true);
  const [mode, setMode] = useState<Mode>("move");
  const [comms, setComms] = useState<any[]>([]);
  const [customOrderOpen, setCustomOrderOpen] = useState(false);
  const [mapLoading, setMapLoading] = useState(true);
  const [loadProgress, setLoadProgress] = useState(0);
  const [diagOpen, setDiagOpen] = useState(false);
  const [diag, setDiag] = useState<any>(null);
  const [directive, setDirective] = useState("");
  const [orderPick, setOrderPick] = useState<string | null>(null);
  const [focusPickMode, setFocusPickMode] = useState(false);
  const [ramCast, setRamCast] = useState<{
    abilityId: string;
    phase: "pick_unit" | "pick_hex";
    sourceUnitId?: string;
  } | null>(null);
  const modeRef = useRef<Mode>("move");
  const ramCastRef = useRef(ramCast);
  const controlPhaseSeenRef = useRef(false);
  const isPhone = useIsPhone();
  modeRef.current = mode;
  ramCastRef.current = ramCast;

  function advanceCommanderMode(actions: any) {
    setMode(nextCommanderMode(actions));
  }

  function selectUnit(id: string, opts?: { userInitiated?: boolean }) {
    const cast = ramCastRef.current;
    if (opts?.userInitiated && modeRef.current === "ram" && cast?.phase === "pick_unit") {
      void handleRamUnitPick(id);
      return;
    }
    setSelected(id);
    if (opts?.userInitiated) {
      setFollowActive(false);
      const u = [...(battleRef.current.friendly_units || []), ...(battleRef.current.opposition_units || [])].find(
        (x: any) => x.unit_instance_id === id
      );
      playUnitSelectAck(u, battleRef.current?.commander_avatar);
    }
  }

  async function castRam(
    abilityId: string,
    extras?: { target_unit_id?: string; q?: number; r?: number }
  ) {
    setBusy(true);
    try {
      playSfx("ram_ability");
      const res = await api.commanderRam(
        battleRef.current.battle_id,
        battleRef.current.state_version,
        abilityId,
        extras
      );
      await playStepFeedback(res, "Drone Commander", "friendly");
      applyBattle(res.battle);
      setRamCast(null);
      setResolveCue(`${abilityId.replaceAll("_", " ")} resolved`);
      advanceCommanderMode(res.battle?.actions);
      setTimeout(() => setResolveCue(null), 1600);
    } catch (e: any) {
      if (!handleConflict(e)) onError(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleRamUnitPick(unitId: string) {
    const cast = ramCastRef.current;
    if (!cast) return;
    const current = battleRef.current;
    const units = [...(current.friendly_units || []), ...(current.opposition_units || [])];
    const u = units.find((x: any) => x.unit_instance_id === unitId);
    if (!u || u.is_decoy) {
      onError("Pick a valid unit for this ability.");
      return;
    }
    if (cast.abilityId === "spoof_unit_location") {
      if (u.side !== "friendly" || u.category === "commander") {
        onError("Spoof needs a friendly squad or drone in signal.");
        return;
      }
      if (u.signal_state === "out_of_signal") {
        onError("That unit is out of signal.");
        return;
      }
      setSelected(unitId);
      setRamCast({ abilityId: cast.abilityId, phase: "pick_hex", sourceUnitId: unitId });
      setResolveCue("Now click an empty hex inside your signal ring.");
      return;
    }
    if (cast.abilityId === "airstrike") {
      if (u.side !== "opposition") {
        onError("Airstrike needs a painted opposition target.");
        return;
      }
      const statuses: string[] = u.statuses || [];
      if (!statuses.includes("painted")) {
        onError("Airstrike requires a painted target — issue Paint Target on your Rangers first.");
        return;
      }
      await castRam(cast.abilityId, { target_unit_id: unitId });
      return;
    }
    if (cast.abilityId === "signal_jamming") {
      if (u.side !== "opposition" || u.category !== "drone") {
        onError("Signal Jamming needs an opposition drone in signal.");
        return;
      }
      await castRam(cast.abilityId, { target_unit_id: unitId });
    }
  }

  battleRef.current = battle;
  busyRef.current = busy;
  followActiveRef.current = followActive;

  async function allocateRamToDrone(droneId: string) {
    if (busyRef.current) return;
    const before = battleRef.current;
    if (!before?.control_phase?.active) return;
    setBusy(true);
    try {
      const res = await api.controlPhaseAllocate(before.battle_id, before.state_version, droneId);
      applyBattle(res.battle);
      setResolveCue("RAM allocated");
    } catch (err: any) {
      if (!handleConflict(err)) setResolveCue(err?.message || "Allocate failed");
    } finally {
      setBusy(false);
    }
  }

  async function reclaimRamFromDrone(droneId: string) {
    if (busyRef.current) return;
    const before = battleRef.current;
    if (!before?.control_phase?.active) return;
    setBusy(true);
    try {
      const res = await api.controlPhaseReclaim(before.battle_id, before.state_version, droneId);
      applyBattle(res.battle);
      setResolveCue("RAM reclaimed");
    } catch (err: any) {
      if (!handleConflict(err)) setResolveCue(err?.message || "Reclaim failed");
    } finally {
      setBusy(false);
    }
  }

  async function finishControlPhase() {
    if (busyRef.current) return;
    const before = battleRef.current;
    if (!before?.control_phase?.active) return;
    setBusy(true);
    try {
      const res = await api.controlPhaseComplete(before.battle_id, before.state_version);
      applyBattle(res.battle);
      setResolveCue("Control Phase complete — commander turn");
      setMode(nextCommanderMode(res.battle?.actions));
    } catch (err: any) {
      if (!handleConflict(err)) setResolveCue(err?.message || "Complete failed");
    } finally {
      setBusy(false);
    }
  }

  function applyBattle(next: any) {
    battleRef.current = next;
    onUpdate(next);
    if (followActiveRef.current && next.active_actor_id) {
      setSelected(next.active_actor_id);
    }
  }

  function handleConflict(e: unknown, opts?: { quiet?: boolean }): boolean {
    if (e instanceof ApiError && e.status === 409) {
      const snap = e.body?.detail?.battle || e.body?.battle;
      if (snap) {
        applyBattle(snap);
        if (!opts?.quiet) onError("Board synced — prior command was stale.");
        return true;
      }
    }
    return false;
  }

  function joinHexPath(a: { q: number; r: number }[], b: { q: number; r: number }[]) {
    if (!a.length) return [...b];
    if (!b.length) return [...a];
    const last = a[a.length - 1];
    const first = b[0];
    if (last.q === first.q && last.r === first.r) return [...a, ...b.slice(1)];
    return [...a, ...b];
  }

  function coalesceMovedEvent(events: any[]) {
    const moves = events.filter((e) => e.type === "unit_moved");
    if (!moves.length) return null;
    if (moves.length === 1) return moves[0];
    const merged = { ...moves[0], payload: { ...(moves[0].payload || {}) } };
    for (const extra of moves.slice(1)) {
      const ep = extra.payload || {};
      merged.payload.path = joinHexPath(merged.payload.path || [], ep.path || []);
      merged.payload.to = ep.to || merged.payload.to;
      const byId = new Map<string, { path: { q: number; r: number }[]; to?: { q: number; r: number }; model_id?: string }>(
        (merged.payload.model_paths || []).map((mp: any) => [
          String(mp.model_id),
          { ...mp, path: [...(mp.path || [])] },
        ]),
      );
      for (const mp of ep.model_paths || []) {
        const id = String(mp.model_id);
        const cur = byId.get(id);
        if (cur) {
          cur.path = joinHexPath(cur.path || [], mp.path || []);
          cur.to = mp.to || cur.to;
        } else {
          byId.set(id, { ...mp, path: [...(mp.path || [])] });
        }
      }
      merged.payload.model_paths = [...byId.values()];
      merged.payload.animation = {
        ...(merged.payload.animation || {}),
        path: merged.payload.path,
        model_paths: merged.payload.model_paths,
      };
    }
    return merged;
  }

  async function playStepFeedback(res: any, fallbackName?: string, fallbackSide?: string) {
    const host = pixiRef.current;
    if (!host) return;
    const events: any[] = res.step_events || [];
    const name = res.actor_name || fallbackName || "Unit";
    const side = res.actor_side || fallbackSide || "friendly";
    const actorId = res.actor_id;

    const moved = coalesceMovedEvent(events);
    const fired = events.find((e) => e.type === "weapon_fired");
    const mineDeployed = events.find((e) => e.type === "mine_deployed");
    const mineTriggered = events.find((e) => e.type === "mine_triggered");
    const held = events.find((e) => e.type === "activation_skipped" || e.payload?.reason === "hold");
    const ramBuffsEarly = events.filter((e) => e.type === "status_applied" && e.payload?.status);
    const ramOnly =
      !moved &&
      !fired &&
      !mineDeployed &&
      !mineTriggered &&
      !held &&
      (ramBuffsEarly.length > 0 ||
        events.some(
          (e) =>
            e.type === "action_spent" &&
            e.payload?.for &&
            !["move", "attack", "paint_target", "self_destruct", "deploy_mine"].includes(String(e.payload.for))
        ));

    // RAM casts should not yank the camera; moves/attacks may recenter.
    if (actorId && !ramOnly) {
      host.ensurePlayZoom(actorId);
    }
    const unitsFromRes = res.battle
      ? [...(res.battle.friendly_units || []), ...(res.battle.opposition_units || [])]
      : [];
    const unitsFromRef = [
      ...(battleRef.current.friendly_units || []),
      ...(battleRef.current.opposition_units || []),
    ];
    const unitBefore = actorId
      ? unitsFromRef.find((u: any) => u.unit_instance_id === actorId)
      : null;
    const unitAfter = actorId
      ? unitsFromRes.find((u: any) => u.unit_instance_id === actorId)
      : null;
    const unitSnap = unitBefore || unitAfter;

    if (moved) {
      const path = (moved.payload?.path || moved.payload?.animation?.path || []) as { q: number; r: number }[];
      const modelPaths = (moved.payload?.model_paths || moved.payload?.animation?.model_paths || []) as {
        model_id: string;
        path: { q: number; r: number }[];
        to?: { q: number; r: number };
      }[];
      const to = moved.payload?.to;
      const fullPath =
        path.length >= 2
          ? path
          : path.length === 1 && to
            ? [...path, to]
            : to
              ? [to]
              : path;
      const isAiming = moved.payload?.reason === "aiming_reposition" || moved.payload?.animation?.type === "squad_aim_move";
      setResolveCue(isAiming ? `${name} is taking aim…` : `${name} is moving…`);
      playSfx("move_confirm");
      const uid = actorId || moved.actor_id || moved.payload?.unit_instance_id;
      let pathsForAnim = modelPaths.length
        ? modelPaths.map((mp) => ({
            model_id: String(mp.model_id || "m1"),
            path: mp.path?.length ? mp.path : mp.to ? [mp.to] : fullPath,
            to: mp.to,
          }))
        : undefined;
      if ((!pathsForAnim || pathsForAnim.length < 2) && unitSnap?.category === "soldier_squad" && fullPath.length >= 2) {
        const living = (unitSnap.models || []).filter((m: any) => m.alive && m.position);
        if (living.length) {
          pathsForAnim = living.map((m: any) => ({
            model_id: String(m.model_id),
            path: fullPath,
            to: fullPath[fullPath.length - 1],
          }));
        }
      }
      await host.playMoveAnimation({
        unitId: uid,
        path: fullPath,
        side,
        label: name,
        assetSetId: unitSnap?.asset_set_id,
        definitionId: unitSnap?.definition_id,
        category: unitSnap?.category,
        commanderAvatar: battleRef.current?.commander_avatar,
        ammo: unitSnap?.ammo,
        modelPaths: pathsForAnim,
      });

      // Move + Self-destruct (or other AOE) in the same resolve step — play blast AFTER the dash lands.
      const aoeAfterMove = events.find((e) => e.payload?.aoe && e.payload?.impact);
      const selfDestructed = events.some(
        (e) =>
          (e.type === "action_spent" && e.payload?.for === "self_destruct") ||
          (e.type === "unit_defeated" && e.payload?.reason === "self_destruct"),
      );
      if ((aoeAfterMove || selfDestructed) && host.playCombatFx) {
        const impact =
          (aoeAfterMove?.payload?.impact as { q: number; r: number } | undefined) ||
          (fullPath.length ? fullPath[fullPath.length - 1] : null) ||
          (to as { q: number; r: number } | undefined);
        const damageHexes = events
          .filter((e) => e.type === "damage_applied")
          .map((e) => e.payload?.animation?.at)
          .filter(Boolean) as { q: number; r: number }[];
        setResolveCue(`${name} detonates!`);
        await host.playCombatFx({
          kind: "explosion_large",
          hits: damageHexes,
          blast: impact,
          weaponId: aoeAfterMove?.payload?.weapon_id || "self_destruct",
        });
        if (res.battle) await host.hydrate(res.battle);
        return;
      }

      // Aiming reposition is followed by the volley in the same step — don't return early
      if (!isAiming || !fired) {
        return;
      }
    }

    if (mineDeployed && host.playDeployMine) {
      const to = mineDeployed.payload?.position || mineDeployed.payload?.animation?.to;
      setResolveCue(`${name} is deploying a mine…`);
      playSfx("move_confirm");
      if (to) {
        await host.playDeployMine({
          unitId: actorId || mineDeployed.actor_id,
          to,
          side,
          assetSetId: unitSnap?.asset_set_id,
          definitionId: unitSnap?.definition_id,
          category: unitSnap?.category,
        });
      }
      return;
    }

    if (mineTriggered && host.playCombatFx) {
      const at = mineTriggered.payload?.position || mineTriggered.payload?.animation?.at;
      const damageHexes = events
        .filter((e) => e.type === "damage_applied")
        .map((e) => e.payload?.animation?.at)
        .filter(Boolean) as { q: number; r: number }[];
      setResolveCue(`Mine detonates!`);
      await host.playCombatFx({
        kind: "explosion_large",
        hits: damageHexes,
        blast: at,
        weaponId: "anti_personnel_mine",
      });
      return;
    }

    if (fired) {
      const anim = fired.payload?.animation || {};
      const shots = (fired.payload?.shots || anim.shots || []) as {
        from: { q: number; r: number };
        to: { q: number; r: number };
        hit?: boolean;
      }[];
      const resolvedHits = new Set(
        events
          .filter((e) => e.type === "attack_resolved" && e.payload?.hit === true)
          .map((e) => `${e.payload?.animation?.to?.q},${e.payload?.animation?.to?.r}`),
      );
      const damageHexes = events
        .filter((e) => e.type === "damage_applied")
        .map((e) => e.payload?.animation?.at)
        .filter(Boolean) as { q: number; r: number }[];
      const aoeImpact = events.find((e) => e.payload?.impact)?.payload?.impact as { q: number; r: number } | undefined;
      const kind = combatKindFromEvents(events);
      setResolveCue(kind === "gunfire" ? `${name} is firing!` : `${name} detonates!`);
      const shotList = shots.map((s) => ({
        from: s.from,
        to: s.to,
        hit: s.hit ?? (resolvedHits.size ? resolvedHits.has(`${s.to?.q},${s.to?.r}`) : true),
      }));
      if (host.playCombatFx) {
        await host.playCombatFx({
          shots: shotList.length
            ? shotList
            : anim.from && anim.to
              ? [{ from: anim.from, to: anim.to, hit: true }]
              : [],
          kind,
          hits: damageHexes,
          blast: aoeImpact || (kind !== "gunfire" ? shots[0]?.to || anim.to : null),
          weaponId: fired.payload?.weapon_id,
          actorCategory: unitSnap?.category,
          commanderAvatar: battleRef.current?.commander_avatar,
          actorSide: side,
          actorAssetSetId: unitSnap?.asset_set_id,
          actorDefinitionId: unitSnap?.definition_id,
        });
      } else if (shots.length > 1 && host.playSquadVolley) {
        playSfx(
          fireSfxForAttack({
            weaponId: fired.payload?.weapon_id,
            category: unitSnap?.category,
            commanderAvatar: battleRef.current?.commander_avatar,
            actorSide: side,
          }),
        );
        await host.playSquadVolley(shots.map((s) => ({ from: s.from, to: s.to })));
        playSfx("attack_impact");
      } else if (anim.from && anim.to) {
        playSfx(
          fireSfxForAttack({
            weaponId: fired.payload?.weapon_id,
            category: unitSnap?.category,
            commanderAvatar: battleRef.current?.commander_avatar,
            actorSide: side,
          }),
        );
        await host.playAttackFlash(anim.from, anim.to);
        playSfx("attack_impact");
      } else if (shots.length === 1) {
        playSfx(
          fireSfxForAttack({
            weaponId: fired.payload?.weapon_id,
            category: unitSnap?.category,
            commanderAvatar: battleRef.current?.commander_avatar,
            actorSide: side,
          }),
        );
        await host.playAttackFlash(shots[0].from, shots[0].to);
        playSfx("attack_impact");
      } else {
        await new Promise((r) => setTimeout(r, 180));
      }
      if (res.battle) await host.hydrate(res.battle);
      return;
    }

    const aoeOnly = events.find((e) => e.payload?.aoe && e.payload?.impact);
    if (aoeOnly && host.playCombatFx) {
      const kind = combatKindFromEvents(events);
      const impact = aoeOnly.payload.impact as { q: number; r: number };
      const damageHexes = events
        .filter((e) => e.type === "damage_applied")
        .map((e) => e.payload?.animation?.at)
        .filter(Boolean) as { q: number; r: number }[];
      setResolveCue(`${name} detonates!`);
      await host.playCombatFx({ kind, hits: damageHexes, blast: impact });
      if (res.battle) await host.hydrate(res.battle);
      return;
    }

    const ramSpent = events.find((e) => e.type === "resource_changed" && e.payload?.resource === "ram" && e.payload?.remaining != null);
    if (ramSpent && ramSpent.payload?.reason !== "round_refresh") {
      setResolveCue(`${name} spent RAM (${ramSpent.payload.remaining} left)`);
      await new Promise((r) => setTimeout(r, 700));
      if (res.battle) await host.hydrate(res.battle);
      return;
    }

    const ramRefresh = events.find(
      (e) => e.type === "resource_changed" && e.payload?.resource === "ram" && e.payload?.reason === "round_refresh",
    );
    if (ramRefresh) {
      setResolveCue(`RAM restored (${ramRefresh.payload.remaining}/${ramRefresh.payload.capacity ?? battleRef.current?.commander?.ram_capacity ?? 6})`);
      await new Promise((r) => setTimeout(r, 650));
      if (res.battle) await host.hydrate(res.battle);
      return;
    }

    const ramBuffs = events.filter((e) => e.type === "status_applied" && e.payload?.status);
    if (ramBuffs.length) {
      const labels = ramBuffs
        .map((e) => String(e.payload?.status || "").replaceAll("_", " "))
        .filter(Boolean);
      setResolveCue(labels.length ? `${name}: ${labels[0]} applied` : `${name} RAM ability resolved`);
      await new Promise((r) => setTimeout(r, 750));
      if (res.battle) await host.hydrate(res.battle);
      return;
    }

    if (held) {
      setResolveCue(`${name} holds position`);
      await new Promise((r) => setTimeout(r, 450));
      return;
    }

    setResolveCue(`${name} acted`);
    await new Promise((r) => setTimeout(r, 350));
  }

  async function resolveAgentsFrom(start: any) {
    let current = start;
    battleRef.current = start;
    setResolving(true);
    resolvingRef.current = true;
    setFollowActive(true);
    try {
      let guard = 0;
      // Focus the unit that is about to act (already active after end-activation)
      if (current.active_actor_id) {
        const units = [...(current.friendly_units || []), ...(current.opposition_units || [])];
        const upcoming = units.find((u: any) => u.unit_instance_id === current.active_actor_id);
        setResolveCue(`Resolving ${upcoming?.display_name || "next unit"}…`);
        setSelected(current.active_actor_id);
        pixiRef.current?.centerOnUnit(current.active_actor_id);
        await new Promise((r) => setTimeout(r, 280));
      }

      while (needsAgentResolve(battleRef.current) && guard < 40) {
        guard += 1;
        const snap = battleRef.current;
        const units = [...(snap.friendly_units || []), ...(snap.opposition_units || [])];
        const upcoming = units.find((u: any) => u.unit_instance_id === snap.active_actor_id);
        setResolveCue(`Resolving ${upcoming?.display_name || "unit"}…`);
        setSelected(snap.active_actor_id);
        if (snap.active_actor_id) pixiRef.current?.centerOnUnit(snap.active_actor_id);

        try {
          const res = await api.resolveNext(snap.battle_id, snap.state_version);
          await playStepFeedback(res, upcoming?.display_name, upcoming?.side);
          current = res.battle;
          applyBattle(current);
          await new Promise((r) => setTimeout(r, 220));
          if (!res.resolved && !needsAgentResolve(battleRef.current)) break;
        } catch (e: any) {
          if (handleConflict(e, { quiet: true })) {
            guard -= 1;
            continue;
          }
          if (!handleConflict(e)) onError(e.message || String(e));
          break;
        }
      }
      setResolveCue(null);
    } catch (e: any) {
      if (!handleConflict(e, { quiet: resolvingRef.current })) onError(e.message || String(e));
    } finally {
      resolvingRef.current = false;
      setResolving(false);
      setResolveCue(null);
    }
    return battleRef.current;
  }

  useEffect(() => {
    startBattleMusic();
    return () => stopBattleMusic();
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!hostRef.current) return;
      const host = await createBattlefield(hostRef.current, {
        onLoadProgress: (pct) => {
          if (!cancelled) setLoadProgress(pct);
        },
        onAssetsReady: () => {
          if (!cancelled) {
            setLoadProgress(100);
            setMapLoading(false);
          }
        },
        onUnitSelected: (id) => {
          selectUnit(id, { userInitiated: true });
        },
        onIntroScan: (playing) => {
          setIntroScanPlaying(playing);
          if (playing) {
            setResolveCue("Hex scan in progress…");
            setBusy(true);
          } else {
            setResolveCue(null);
            setBusy(false);
            // Belt-and-suspenders: intro end can race React control-phase effect on round 1.
            const current = battleRef.current;
            const cp = current?.control_phase;
            if (
              window.matchMedia("(max-width: 480px)").matches &&
              cp?.active &&
              cp?.side === "friendly"
            ) {
              const cmdId =
                cp.commander_id ||
                current?.commander?.unit_instance_id ||
                current?.active_actor_id;
              if (cmdId) {
                requestAnimationFrame(() => {
                  requestAnimationFrame(() => pixiRef.current?.zoomToMaxOnUnit(cmdId));
                });
              }
            }
          }
        },
        onHexSelected: async (hex) => {
          if (busyRef.current) return;
          const current = battleRef.current;
          if (current?.control_phase?.active) return;
          const units = [...(current.friendly_units || []), ...(current.opposition_units || [])];
          const active = units.find((u: any) => u.unit_instance_id === current.active_actor_id);
          if (!isFriendlyCommander(active)) return;
          const cast = ramCastRef.current;
          if (modeRef.current === "ram" && cast?.phase === "pick_hex" && cast.sourceUnitId) {
            await castRam(cast.abilityId, {
              target_unit_id: cast.sourceUnitId,
              q: hex.q,
              r: hex.r,
            });
            return;
          }
          // Actions may be taken in any order — hex clicks move whenever Move remains,
          // even if the Attack/RAM/Orders tab is still selected.
          if (cast?.phase === "pick_unit") return;
          const actions = current.actions || {};
          const movesSpent = Number(actions.moves_spent || 0);
          // Second move spends Attack; after two moves, no more movement.
          const canMove =
            movesSpent < 2 && ((actions.move || 0) > 0 || (actions.standard || 0) > 0);
          if (!canMove) return;
          const hit = (current.reachable_hexes || []).some((h: any) => h.q === hex.q && h.r === hex.r);
          if (!hit) return;
          if (modeRef.current !== "move") setMode("move");
          await moveTo(hex.q, hex.r);
        },
        onRamAllocate: (droneId) => {
          void allocateRamToDrone(droneId);
        },
        onRamReclaim: (droneId) => {
          void reclaimRamFromDrone(droneId);
        },
      });
      if (cancelled) {
        host.destroy();
        return;
      }
      pixiRef.current = host;
      await host.hydrate(battleRef.current);
    })().catch((e) => {
      if (!cancelled) {
        setMapLoading(false);
        onError(e?.message || String(e));
      }
    });
    return () => {
      cancelled = true;
      pixiRef.current?.destroy();
      pixiRef.current = null;
    };
  }, []);

  useEffect(() => {
    void pixiRef.current?.hydrate(battle);
    const units = [...(battle.friendly_units || []), ...(battle.opposition_units || [])];
    const activeNow = units.find((u: any) => u.unit_instance_id === battle.active_actor_id);
    const commanderTurn = isFriendlyCommander(activeNow);
    pixiRef.current?.setInteractable(!busy && !resolving && commanderTurn);
    if (followActive && battle.active_actor_id) setSelected(battle.active_actor_id);
    pixiRef.current?.setSelection(selected);

    // Spoof placement: highlight empty hexes in signal range (not move range)
    if (ramCast?.phase === "pick_hex" && commanderTurn) {
      const cmd = battle.commander || units.find((u: any) => u.category === "commander" && u.side === "friendly");
      const radius = Number(battle.signal_radius || 0);
      const width = Number(battle.map?.width || 50);
      const height = Number(battle.map?.height || 50);
      if (cmd?.position && radius > 0) {
        const occupied = new Set(
          units.filter((u: any) => u.alive && !u.is_decoy).map((u: any) => `${u.position.q},${u.position.r}`)
        );
        const cq = cmd.position.q;
        const cr = cmd.position.r;
        const hexes: { q: number; r: number }[] = [];
        for (let q = 0; q < width; q++) {
          for (let r = 0; r < height; r++) {
            const dq = q - cq;
            const dr = r - cr;
            const dist = (Math.abs(dq) + Math.abs(dq + dr) + Math.abs(dr)) / 2;
            if (dist > radius) continue;
            if (dist === 0) continue;
            if (occupied.has(`${q},${r}`)) continue;
            hexes.push({ q, r });
          }
        }
        pixiRef.current?.setPlacementHexes(hexes);
      } else {
        pixiRef.current?.setPlacementHexes(null);
      }
    } else {
      pixiRef.current?.setPlacementHexes(null);
    }
  }, [battle, busy, resolving, followActive, selected, ramCast]);

  useEffect(() => {
    api.communications(battle.battle_id)
      .then((r) => setComms(r.entries || []))
      .catch(() => {});
  }, [battle.battle_id, battle.state_version]);

  // Fresh commander activation → start on the best available action tab.
  useEffect(() => {
    const units = [...(battle.friendly_units || []), ...(battle.opposition_units || [])];
    const activeNow = units.find((u: any) => u.unit_instance_id === battle.active_actor_id);
    if (isFriendlyCommander(activeNow)) {
      setMode(nextCommanderMode(battle.actions || {}));
      setRamCast(null);
    }
  }, [battle.active_actor_id]);

  async function moveTo(q: number, r: number) {
    setBusy(true);
    try {
      playSfx("hex_select");
      const before = battleRef.current;
      const res = await api.commanderMove(before.battle_id, before.state_version, q, r);
      await playStepFeedback(
        { ...res, actor_id: before.active_actor_id, actor_name: "Drone Commander", actor_side: "friendly" },
        "Drone Commander",
        "friendly"
      );
      applyBattle(res.battle);
      advanceCommanderMode(res.battle?.actions);
      // Stay at playable zoom after a commander move — never snap back to overview.
      pixiRef.current?.ensurePlayZoom(before.active_actor_id);
    } catch (e: any) {
      if (!handleConflict(e)) onError(e.message || String(e));
    } finally {
      setBusy(false);
      setResolveCue(null);
    }
  }

  function tryRamAbility(opt: any) {
    if (busy || resolving || ramCast) return;
    const blocked = opt?.preview?.blocked_reason;
    if (opt?.preview?.disabled || blocked) {
      setResolveCue(blocked || "Ability not available");
      return;
    }
    void act(opt.option_id);
  }

  async function act(optionId: string) {
    setBusy(true);
    try {
      const opt = (battleRef.current.legal_player_options || []).find((o: any) => o.option_id === optionId);
      const sub = opt?.subroutine;
      if (sub === "ram_ability" && opt?.preview?.needs_target) {
        const aid = opt.preview.ability_id as string;
        setMode("ram");
        setRamCast({ abilityId: aid, phase: "pick_unit" });
        setResolveCue(
          aid === "spoof_unit_location"
            ? "Spoof: select a friendly unit in signal, then an empty hex."
            : aid === "airstrike"
              ? "Airstrike: select a painted opposition unit."
              : aid === "signal_jamming"
                ? "Jam: select an opposition drone inside your signal ring."
                : `Select a target for ${opt.label}`
        );
        setBusy(false);
        return;
      }
      if (sub === "ram_ability" && !opt?.preview?.needs_target) {
        await castRam(String(opt?.preview?.ability_id || optionId.replace(/^ram:/, "")));
        return;
      }
      const res = await api.commanderAction(battleRef.current.battle_id, battleRef.current.state_version, optionId);
      if (sub === "attack" || sub === "self_destruct") {
        await playStepFeedback(res, "Drone Commander", "friendly");
      } else if (sub === "ram_ability") {
        playSfx("ram_ability");
        await playStepFeedback(res, "Drone Commander", "friendly");
      }
      let next = res.battle;
      applyBattle(next);
      setRamCast(null);
      advanceCommanderMode(next?.actions);
      if (needsAgentResolve(next)) {
        next = await resolveAgentsFrom(next);
      }
    } catch (e: any) {
      if (!handleConflict(e)) onError(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  async function endTurn() {
    setBusy(true);
    try {
      const res = await api.endActivation(battleRef.current.battle_id, battleRef.current.state_version);
      let next = res.battle;
      applyBattle(next);
      setFollowActive(true);
      setCustomOrderOpen(false);
      setMode("move");
      if (needsAgentResolve(next)) {
        next = await resolveAgentsFrom(next);
      }
    } catch (e: any) {
      if (!handleConflict(e)) onError(e.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  async function issueArmyOrder(
    orderId: string,
    text = "",
    targetRefs: Array<{ kind: string; unit_instance_id: string }> = []
  ) {
    const snap = battleRef.current;
    if (!snap?.battle_id) return;
    try {
      const next = await api.setDirective(snap.battle_id, text, null, {
        order_id: orderId,
        target_refs: targetRefs,
      });
      applyBattle(next);
      setDirective("");
      setOrderPick(null);
      setFocusPickMode(false);
      if (orderId !== "custom") playArmyOrderVo(orderId, battleRef.current?.commander_avatar);
      else playSfx("directive");
      const label =
        orderId === "custom"
          ? text.trim()
          : boot?.army_orders?.find((o: any) => o.id === orderId)?.label || orderId;
      if (label) {
        setResolveCue(resolvingRef.current ? `Order updated: ${label}` : `Army order: ${label}`);
      }
    } catch (e: any) {
      if (!handleConflict(e, { quiet: resolvingRef.current })) onError(e.message || String(e));
    }
  }

  async function queueDirective() {
    if (!directive.trim()) return;
    await issueArmyOrder("custom", directive.trim());
    setCustomOrderOpen(false);
    setOrderPick(null);
  }

  function openCustomOrder() {
    setFocusPickMode(false);
    setOrderPick("custom");
    if (isPhone) setCustomOrderOpen(true);
  }

  const allUnits = useMemo(
    () => [...(battle.friendly_units || []), ...(battle.opposition_units || [])],
    [battle]
  );
  const active = allUnits.find((u: any) => u.unit_instance_id === battle.active_actor_id);
  const selectedUnit = allUnits.find((u: any) => u.unit_instance_id === selected) || active;
  const isCommanderTurn = isFriendlyCommander(active);
  const controlPhaseActive = !!(
    battle.control_phase?.active &&
    battle.control_phase?.side === "friendly" &&
    isCommanderTurn &&
    !introScanPlaying
  );
  const commanderActionsOpen = isCommanderTurn && !controlPhaseActive;
  /** Army orders are radio traffic — issue anytime during live battle, including during unit autonomy. */
  const armyOrdersEnabled = battle.status === "ACTIVE" && !introScanPlaying;

  useEffect(() => {
    if (controlPhaseActive && !controlPhaseSeenRef.current) {
      const cmdId =
        battle.control_phase?.commander_id ||
        battle.commander?.unit_instance_id ||
        battle.active_actor_id;
      if (isPhone && cmdId) {
        pixiRef.current?.zoomToMaxOnUnit(cmdId);
      }
    }
    controlPhaseSeenRef.current = controlPhaseActive;
  }, [controlPhaseActive, isPhone, battle.control_phase?.commander_id, battle.commander?.unit_instance_id, battle.active_actor_id]);

  const options = battle.legal_player_options || [];
  const attackOpts = options.filter(
    (o: any) => o.subroutine === "attack" || o.subroutine === "self_destruct" || o.subroutine === "paint_target"
  );
  const ramOpts = options.filter((o: any) => o.subroutine === "ram_ability");

  const actions = battle.actions || { standard: 0, move: 0, minor: 0, moves_spent: 0 };
  const flags = readActionFlags(actions);
  const secondMoveCostsAttack = !!battle.second_move_costs_attack;

  const armyOrders = (boot?.army_orders || []).filter((o: any) => o.id !== "custom");
  const armyOrderCustom = (boot?.army_orders || []).find((o: any) => o.id === "custom");
  const currentArmyOrder = (battle.directives || []).find((d: any) => d.scope === "global" && d.active);
  const oppositionTargets = (battle.opposition_units || []).filter((u: any) => u.alive !== false);

  const portrait = selectedUnit ? unitPortraitUrl(selectedUnit, battle.commander_avatar) : null;

  const objectiveLabel = battle.objective?.label || "Objective";
  const objectiveControl = battle.objective?.control || "empty";

  const mapHost = (
    <div className="battlefield-host" ref={hostRef} role="img" aria-label="Tactical battlefield" />
  );

  return (
    <div className={`battle-layout sc-hud ${isPhone ? "phone-battle" : ""}`}>
      {mapLoading && (
        <div className="battle-load-overlay" role="status" aria-live="polite" aria-busy="true">
          <img
            className="battle-load-logo"
            src="/assets/ui/battle-loading-logo.png"
            alt="Drone Commander"
          />
          <div className="battle-load-bar" aria-hidden="true">
            <div className="battle-load-bar-fill" style={{ width: `${loadProgress}%` }} />
          </div>
          <span className="visually-hidden">Loading battlefield… {loadProgress}%</span>
        </div>
      )}
      <div className="battle-top row">
        <strong className="battle-round">R{battle.round}</strong>
        <span className="objective">
          {isPhone ? (
            <>
              {battle.objective?.friendly_vp ?? 0}–{battle.objective?.opposition_vp ?? 0}/{battle.objective?.vp_to_win ?? 5}
              {objectiveControl === "contested"
                ? " · c"
                : objectiveControl === "friendly"
                  ? " · blue"
                  : objectiveControl === "opposition"
                    ? " · red"
                    : ""}
            </>
          ) : (
            <>
              {objectiveLabel} {battle.objective?.friendly_vp ?? 0}–{battle.objective?.opposition_vp ?? 0} /{" "}
              {battle.objective?.vp_to_win ?? 5}
              {objectiveControl === "contested"
                ? " · contested"
                : objectiveControl === "friendly"
                  ? " · blue control"
                  : objectiveControl === "opposition"
                    ? " · red control"
                    : " · empty"}
            </>
          )}
        </span>
        <span className={battle.status === "ACTIVE" ? "live" : "warn"}>{isPhone ? (battle.status === "ACTIVE" ? "LIVE" : battle.status) : battle.status}</span>
        {!isPhone && (
          <span className="muted">
            Active: <strong>{active?.display_name || "—"}</strong>
            {active ? ` (${active.side})` : ""}
          </span>
        )}
        {isCommanderTurn ? (
          <span className="action-tokens" aria-label="Remaining actions">
            <span className={actions.standard ? "on" : "off"} title="Standard / Attack">
              S {actions.standard}
            </span>
            <span className={actions.move ? "on" : "off"} title="Move">
              M {actions.move}
            </span>
            <span className={actions.minor ? "on" : "off"} title="Minor">
              m {actions.minor}
            </span>
          </span>
        ) : (
          <span className="warn battle-autonomy-cue">{isPhone ? resolveCue || "Autonomy" : resolveCue || "Unit autonomy — pan/inspect only"}</span>
        )}
        {secondMoveCostsAttack && isCommanderTurn && (
          <span className="warn">{isPhone ? "2nd=Atk" : "2nd move costs Attack"}</span>
        )}
        {!isPhone && (
          <span className="legend">
            <span>
              <i className="grid" /> Hex grid
            </span>
            <span>
              <i className="move" /> Reachable
            </span>
            <span>
              <i className="atk" /> Attack
            </span>
            <span>
              <i className="sig" /> Signal
            </span>
          </span>
        )}
        {!followActive && (
          <button type="button" className="ghost battle-follow-btn" onClick={() => setFollowActive(true)}>
            {isPhone ? "Follow" : "Follow active"}
          </button>
        )}
      </div>

      <div className="field-wrap">
        {isPhone ? mapHost : <TabletFrame>{mapHost}</TabletFrame>}
        <div className="camera-controls">
          <button type="button" onClick={() => pixiRef.current?.zoomBy(1.15)} aria-label="Zoom in">
            +
          </button>
          <button type="button" onClick={() => pixiRef.current?.zoomBy(0.87)} aria-label="Zoom out">
            −
          </button>
          <button type="button" onClick={() => pixiRef.current?.ensurePlayZoom(battle.active_actor_id)} aria-label="Center active">
            Home
          </button>
          <button type="button" onClick={() => pixiRef.current?.centerOnMap()} aria-label="Lock whole map">
            Map
          </button>
        </div>
        {(busy || resolving) && (
          <div className="resolving-banner" role="status">
            {resolveCue || (resolving ? `Agents resolving — ${active?.display_name || "next unit"}` : "Working…")}
          </div>
        )}
        {resolveCue && !busy && !resolving && (
          <div className="resolving-banner" role="status">
            {resolveCue}
          </div>
        )}
        {controlPhaseActive && (
          <div
            className={`control-phase-panel ${isPhone ? "control-phase-panel-phone" : ""}`}
            role="dialog"
            aria-label="Control Phase RAM Allocation"
          >
            {!isPhone && (
              <p className="control-phase-hint">
                Control Phase — drag RAM from the commander onto blue-highlighted drones in signal. Small drones (one-ways / dogs / direct-attack) max 1 RAM each; larger drones max 3. Leftover RAM stays for your abilities.
              </p>
            )}
            <button
              type="button"
              className="control-phase-complete"
              disabled={busy || resolving}
              onClick={() => void finishControlPhase()}
            >
              {isPhone ? "RAM Done" : "RAM Allocation Complete"}
            </button>
          </div>
        )}
        {isPhone && customOrderOpen && (
          <div className="custom-order-overlay" role="dialog" aria-label="Custom army order">
            <div className="custom-order-card">
              <strong>Custom Order</strong>
              <textarea
                value={directive}
                onChange={(e) => setDirective(e.target.value)}
                rows={4}
                placeholder="Type your army order…"
                autoFocus
              />
              <div className="row">
                <button
                  type="button"
                  className="primary"
                  disabled={!directive.trim() || !armyOrdersEnabled}
                  onClick={() => void queueDirective()}
                >
                  Issue
                </button>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => {
                    setCustomOrderOpen(false);
                    setOrderPick(null);
                    setDirective("");
                  }}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="bottom-hud">
        <div className="hud-commands stack">
          <div className="row cmd-row commander-modes">
            <button
              className={`mode-btn ${mode === "move" ? "on" : ""} ${flags.moveSpentUi ? "spent" : ""}`}
              disabled={!commanderActionsOpen || busy || resolving}
              onClick={() => {
                setMode("move");
                setRamCast(null);
                setResolveCue(null);
                if (battle.active_actor_id) pixiRef.current?.ensurePlayZoom(battle.active_actor_id);
              }}
            >
              Move
            </button>
            <button
              className={`mode-btn ${mode === "attack" ? "on" : ""} ${flags.attackSpentUi ? "spent" : ""}`}
              disabled={!commanderActionsOpen || busy || resolving}
              onClick={() => {
                setMode("attack");
                setRamCast(null);
                setResolveCue(null);
                if (battle.active_actor_id) pixiRef.current?.ensurePlayZoom(battle.active_actor_id);
              }}
            >
              Attack
            </button>
            <button
              className={`mode-btn ${mode === "ram" ? "on" : ""} ${flags.ramSpentUi ? "spent" : ""}`}
              disabled={!commanderActionsOpen || busy || resolving}
              onClick={() => {
                setMode("ram");
                setRamCast(null);
                setResolveCue(null);
              }}
            >
              RAM
            </button>
            <button
              className={`primary end-activation ${mode === "end" || flags.ramSpentUi ? "end-ready" : ""}`}
              disabled={busy || resolving || !commanderActionsOpen}
              onClick={endTurn}
            >
              End{isPhone ? "" : " Activation"}
            </button>
          </div>

          <div className="commander-actions-body">
            {controlPhaseActive && !isPhone && (
              <p className="muted hint">
                Allocate RAM on the map, then press RAM Allocation Complete (lower left).
              </p>
            )}
            {commanderActionsOpen && mode === "move" && (
              <p className="muted hint">
                {isPhone ? "Tap a green hex to move." : `Click a green hex within Speed to move${secondMoveCostsAttack ? " (another move spends your Attack)." : "."}`}
              </p>
            )}
            {commanderActionsOpen && mode === "attack" && (
              <div className="dest-list attack-list" aria-label="Attack targets">
                {!attackOpts.length && <span className="muted">No targets in range.</span>}
                {attackOpts.map((opt: any) => (
                  <button
                    key={opt.option_id}
                    className="option-btn"
                    data-sfx="combat"
                    disabled={busy || resolving}
                    onClick={() => act(opt.option_id)}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            )}
            {commanderActionsOpen && mode === "ram" && (
              <div className="dest-list ram-list" aria-label="RAM abilities">
                <div className="ram-pool-line live">
                  RAM {selectedUnit?.ram_current ?? battle.commander?.ram_current ?? 0}/
                  {selectedUnit?.ram_capacity ?? battle.commander?.ram_capacity ?? 6}
                </div>
                {ramCast && (
                  <button
                    type="button"
                    className="ghost ram-cancel"
                    onClick={() => {
                      setRamCast(null);
                      setResolveCue(null);
                    }}
                  >
                    Cancel targeting
                  </button>
                )}
                {!ramOpts.length && !ramCast && (
                  <span className="muted">No RAM abilities loaded.</span>
                )}
                {ramOpts.map((opt: any) => {
                  const blocked = !!(opt.preview?.disabled || opt.preview?.blocked_reason);
                  return (
                    <button
                      key={opt.option_id}
                      className={`option-btn ram-ability-btn ${ramCast?.abilityId === opt.preview?.ability_id ? "primary" : ""} ${blocked ? "blocked" : ""}`}
                      disabled={busy || resolving || !!ramCast}
                      onClick={() => tryRamAbility(opt)}
                    >
                      {ramAbilityShortName(opt, boot)}
                    </button>
                  );
                })}
              </div>
            )}
            {commanderActionsOpen && mode === "end" && (
              <p className="muted hint">End Activation to pass the turn.</p>
            )}
            {!isCommanderTurn && (
              <p className="muted hint">
                {armyOrdersEnabled ? "Unit autonomy — shift orders in Radio." : "Unit autonomy."}
              </p>
            )}
          </div>

          <div className="row cmd-row cmd-utils">
            <button
              type="button"
              className="ghost"
              disabled={busy || resolving}
              onClick={async () => {
                try {
                  applyBattle(await api.getBattle(battle.battle_id));
                } catch (e: any) {
                  onError(e.message);
                }
              }}
            >
              Refresh
            </button>
            <button
              type="button"
              className="ghost"
              disabled={busy || resolving}
              onClick={async () => {
                try {
                  const d = await api.diagnosticsBattle(battle.battle_id);
                  setDiag(d);
                  setDiagOpen(true);
                } catch (e: any) {
                  onError(e.message);
                }
              }}
            >
              Diag
            </button>
          </div>

          {diagOpen && diag && !isPhone && (
            <div className="panel diag-panel">
              <div className="row" style={{ justifyContent: "space-between" }}>
                <strong>Diagnostics</strong>
                <button type="button" onClick={() => setDiagOpen(false)}>
                  Close
                </button>
              </div>
              <div className="muted">
                requests {diag.stats?.request_count} · events {diag.stats?.event_count} · agent runs{" "}
                {diag.stats?.agent_run_count} · fallbacks {diag.stats?.fallback_count}
              </div>
              <div className="diag-runs">
                {(diag.agent_runs || diag.recent_agent_runs || []).slice(-6).reverse().map((run: any) => (
                  <div key={run.id} className="muted diag-run">
                    {run.definition_id} · {run.status} · {run.provider || "none"}
                    {run.model ? `/${run.model}` : ""}
                    {run.confidence != null ? ` · conf ${Number(run.confidence).toFixed(2)}` : ""}
                    {run.margin != null ? ` · margin ${Number(run.margin).toFixed(2)}` : ""}
                    {run.provider_latency_ms != null ? ` · ${Math.round(run.provider_latency_ms)} ms` : ""}
                    {run.shadow ? ` · shadow ${run.shadow_agreement ? "agree" : "differ"}` : ""}
                    {run.fallback_reason ? ` · ${String(run.fallback_reason).slice(0, 80)}` : ""}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="radio-cell stack" aria-label="Radio and army orders">
          <div className="radio-cell-header">
            <strong>Radio</strong>
          </div>
          <div className="directive-box orders-panel radio-panel">
            <div className="order-chips stack">
              {armyOrders.map((o: any) => (
                <button
                  key={o.id}
                  type="button"
                  className={
                    orderPick === o.id || (currentArmyOrder?.order_id === o.id && !orderPick) ? "primary" : ""
                  }
                  data-sfx="order"
                  disabled={!armyOrdersEnabled}
                  onClick={() => {
                    if (o.requires_target) {
                      setOrderPick(o.id);
                      setFocusPickMode(true);
                      setCustomOrderOpen(false);
                      return;
                    }
                    void issueArmyOrder(o.id);
                  }}
                >
                  {o.label}
                </button>
              ))}
              <button
                type="button"
                className={orderPick === "custom" ? "primary" : ""}
                data-sfx="order"
                disabled={!armyOrdersEnabled}
                onClick={openCustomOrder}
              >
                {armyOrderCustom?.label || "Custom Order"}
              </button>
            </div>
            {focusPickMode && (
              <div className="order-target-pick" aria-label="Order target picker">
                {!oppositionTargets.length && <span className="muted">No targets.</span>}
                {oppositionTargets.map((u: any) => (
                  <button
                    key={u.unit_instance_id}
                    type="button"
                    className="option-btn"
                    onClick={() =>
                      void issueArmyOrder(orderPick || "focus_fire", "", [
                        { kind: "unit", unit_instance_id: u.unit_instance_id },
                      ])
                    }
                  >
                    {u.display_name}
                  </button>
                ))}
                <button
                  type="button"
                  className="ghost"
                  onClick={() => {
                    setFocusPickMode(false);
                    setOrderPick(null);
                  }}
                >
                  Cancel
                </button>
              </div>
            )}
            {!isPhone && orderPick === "custom" && (
              <>
                <textarea
                  value={directive}
                  onChange={(e) => setDirective(e.target.value)}
                  rows={2}
                  placeholder="Custom army order…"
                />
                <div className="row">
                  <button disabled={!directive.trim() || !armyOrdersEnabled} onClick={() => void queueDirective()}>
                    Issue
                  </button>
                  <button type="button" className="ghost" onClick={() => setOrderPick(null)}>
                    Cancel
                  </button>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="hud-inspect">
          {selectedUnit ? (
            <>
              <div className="portrait-frame">
                {portrait && <img src={portrait} alt={selectedUnit.display_name} />}
              </div>
              <div className="inspect-stats stack">
                <strong>{selectedUnit.display_name}</strong>
                <div className="muted">
                  {selectedUnit.side} · {selectedUnit.category}
                </div>
                <div>
                  HP {selectedUnit.total_hp}
                  {selectedUnit.model_count > 1
                    ? ` · Models ${selectedUnit.living_model_count}/${selectedUnit.model_count}`
                    : ""}
                </div>
                <div>
                  Atk +{selectedUnit.effective_stats.attack} · Def {selectedUnit.effective_stats.defense} · Arm{" "}
                  {selectedUnit.effective_stats.armor} · Spd {selectedUnit.effective_stats.speed}
                </div>
                {selectedUnit.ram_capacity != null && (
                  <div className="live">
                    RAM pool {selectedUnit.ram_current ?? 0}/{selectedUnit.ram_capacity}
                  </div>
                )}
                {selectedUnit.ammo && Object.keys(selectedUnit.ammo).length > 0 && (
                  <div>
                    Ammo{" "}
                    {Object.entries(selectedUnit.ammo)
                      .map(([id, count]) => `${String(id).replaceAll("_", " ")} ${count}`)
                      .join(" · ")}
                  </div>
                )}
                <div className="muted">
                  Signal: {selectedUnit.signal_state} · Hex ({selectedUnit.position.q}, {selectedUnit.position.r})
                </div>
              </div>
            </>
          ) : (
            <div className="muted">Select a unit</div>
          )}
        </div>

        <div className="hud-cards" aria-label="Initiative order">
          {(battle.initiative || []).map((card: any, idx: number) => {
            const u = allUnits.find((x: any) => x.unit_instance_id === card.unit_instance_id);
            const thumb = u ? unitPortraitUrl(u, battle.commander_avatar) : null;
            return (
              <button
                type="button"
                key={card.unit_instance_id}
                className={[
                  "init-card",
                  card.side || "",
                  card.unit_instance_id === battle.active_actor_id ? "active" : "",
                  card.activated ? "done" : "",
                  card.unit_instance_id === selected ? "selected" : "",
                ].join(" ")}
                style={{ zIndex: (battle.initiative?.length || 0) - idx }}
                onClick={() => {
                  selectUnit(card.unit_instance_id, { userInitiated: true });
                }}
              >
                {thumb && <img src={thumb} alt="" />}
                <div className="init-card-meta">
                  <strong>{card.display_name}</strong>
                  <span className="side-tag muted">
                    {card.unit_instance_id === battle.active_actor_id
                      ? "ACTIVE"
                      : card.activated
                        ? "done"
                        : "waiting"}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
