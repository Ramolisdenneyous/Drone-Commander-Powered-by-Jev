from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .agents import provider as llm
from .application import diagnostics as diag
from .application import services as app_services
from .config import settings
from .content.loader import get_catalog, load_catalog
from .db import Base, engine, get_db
from .persistence.models import DomainEventRow, FeedbackRow, SessionRow
from .persistence.schema import ensure_schema
from .telemetry.logging import setup_logging, structured_log
from .telemetry.middleware import ObservabilityMiddleware

setup_logging()

app = FastAPI(title="Drone Commander MK2", version="0.2.0")
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_schema()
    load_catalog(force=True)
    jev_models = llm.refresh_jev_model_catalog()
    structured_log(
        "backend_started",
        content_version=settings.content_version,
        llm_external_enabled=settings.llm_external_enabled,
        tactical_provider=settings.tactical_provider,
        tactical_shadow=settings.tactical_shadow,
        radio_provider=llm.radio_provider_name(),
        jev_models=jev_models.get("models") or [],
        llm_model_tactical=settings.llm_model_tactical,
        log_level=settings.log_level,
        artifact_retention_mode=settings.artifact_retention_mode,
    )


class CommandEnvelope(BaseModel):
    command_id: str = Field(default_factory=lambda: str(uuid4()))
    expected_state_version: int | None = None
    client_content_version: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health(request: Request):
    catalog = get_catalog()
    return {
        "ok": True,
        "service": "drone-commander",
        "content_version": catalog.content_version,
        "llm_external_enabled": settings.llm_external_enabled,
        "tactical_provider": settings.tactical_provider,
        "tactical_provider_configured": (
            bool(settings.typesafe_api_key)
            if settings.tactical_provider.strip().lower() == "jev"
            else bool(settings.openai_api_key)
        ),
        "tactical_shadow": settings.tactical_shadow,
        "radio_provider": llm.radio_provider_name(),
        "typesafe_model": settings.typesafe_model if settings.tactical_provider.strip().lower() == "jev" else None,
        "jev_models": llm.discovered_jev_models().get("models") or [],
        "llm_model_tactical": settings.llm_model_tactical,
        "observability": {
            "structured_logs": True,
            "request_logs": True,
            "agent_runs": True,
            "llm_artifacts": True,
            "outbox": True,
            "artifact_retention_mode": settings.artifact_retention_mode,
            "log_level": settings.log_level,
        },
        "trace_id": getattr(request.state, "trace_id", None),
    }


@app.get("/api/v1/catalog/boot")
def catalog_boot():
    from .engine.scenarios import SCENARIO_META

    catalog = get_catalog()
    scenarios = [
        {
            "id": sid,
            "display_name": meta["display_name"],
            "description": meta["description"],
            "objective_type": meta["objective_type"],
            "default": bool(meta.get("default")),
        }
        for sid, meta in SCENARIO_META.items()
    ]
    return {
        "content_version": catalog.content_version,
        "point_caps": [15, 25, 40, 55, 75, 100],
        "scenarios": scenarios,
        "maps": [
            {
                "id": m.id,
                "display_name": m.display_name,
                "width": m.width,
                "height": m.height,
                "ground_asset": m.ground_asset,
                "select_asset": m.select_asset or m.ground_asset,
            }
            for m in catalog.maps.values()
        ],
        "missions": [m.model_dump() for m in catalog.missions.values()],
        "loadouts": [l.model_dump() for l in catalog.loadouts.values()],
        "units": [
            u.model_dump()
            for u in catalog.units.values()
            if "friendly" in u.side_availability and u.category != "commander"
        ],
        "abilities": [a.model_dump() for a in catalog.abilities.values()],
        "army_orders": [o.model_dump() for o in catalog.army_orders.values()],
        "avatars": [
            {
                "id": key,
                "label": catalog.loadouts[key].display_name,
                "asset": asset,
                "speed": catalog.loadouts[key].speed,
                "attack": catalog.loadouts[key].attack,
                "defense": catalog.loadouts[key].defense,
                "armor": catalog.loadouts[key].armor,
                "hp": catalog.loadouts[key].hp,
                "ram_capacity": catalog.loadouts[key].ram_capacity,
                "passive": catalog.loadouts[key].passive,
                "allowed_abilities": catalog.loadouts[key].allowed_abilities,
            }
            for key, asset in (
                ("male", "/assets/avatars/Drone-commander-Male.png"),
                ("female", "/assets/avatars/Drone-commander-Female.png"),
            )
            if key in catalog.loadouts
        ],
        "asset_manifest": catalog.asset_manifest,
    }


@app.get("/api/v1/catalog")
def catalog_full():
    return catalog_boot()


@app.post("/api/v1/sessions")
def create_session(db: Session = Depends(get_db)):
    return app_services.create_session(db)


@app.get("/api/v1/sessions/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)):
    row = db.get(SessionRow, session_id)
    if not row:
        raise HTTPException(404, "Session not found")
    return app_services.session_snapshot(row)


@app.put("/api/v1/sessions/{session_id}/prep")
def put_prep(session_id: str, body: CommandEnvelope, db: Session = Depends(get_db)):
    row = db.get(SessionRow, session_id)
    if not row:
        raise HTTPException(404, "Session not found")
    if row.status != "DRAFT":
        raise HTTPException(409, "Prep locked")
    return app_services.update_prep(db, row, body.payload, body.command_id)


@app.post("/api/v1/sessions/{session_id}/deploy")
def deploy(session_id: str, body: CommandEnvelope, db: Session = Depends(get_db)):
    row = db.get(SessionRow, session_id)
    if not row:
        raise HTTPException(404, "Session not found")
    try:
        seed = body.payload.get("seed")
        return app_services.deploy(db, row, body.command_id, seed=seed)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/v1/sessions/{session_id}/reset")
def reset_session(session_id: str, db: Session = Depends(get_db)):
    row = db.get(SessionRow, session_id)
    if not row:
        raise HTTPException(404, "Session not found")
    # Create fresh draft generation
    return app_services.create_session(db)


@app.get("/api/v1/battles/{battle_id}")
def get_battle(battle_id: str, db: Session = Depends(get_db)):
    row = db.query(SessionRow).filter(SessionRow.battle_id == battle_id).first()
    if not row:
        raise HTTPException(404, "Battle not found")
    battle = app_services.load_battle_from_row(row) or app_services.get_battle(battle_id)
    if not battle:
        raise HTTPException(404, "Battle not found")
    return app_services.player_snapshot(battle)


@app.get("/api/v1/battles/{battle_id}/events")
def get_events(battle_id: str, after: int = 0, db: Session = Depends(get_db)):
    rows = (
        db.query(DomainEventRow)
        .filter(DomainEventRow.battle_id == battle_id, DomainEventRow.sequence > after)
        .order_by(DomainEventRow.sequence.asc())
        .all()
    )
    return {"events": [r.envelope_json for r in rows]}


@app.get("/api/v1/battles/{battle_id}/stream")
async def stream_events(battle_id: str, last_event_id: int = 0, db: Session = Depends(get_db)):
    async def event_generator():
        cursor = last_event_id
        idle = 0
        while idle < 120:
            rows = (
                db.query(DomainEventRow)
                .filter(DomainEventRow.battle_id == battle_id, DomainEventRow.sequence > cursor)
                .order_by(DomainEventRow.sequence.asc())
                .all()
            )
            if rows:
                idle = 0
                for r in rows:
                    cursor = r.sequence
                    yield f"id: {r.sequence}\ndata: {json.dumps(r.envelope_json)}\n\n"
            else:
                idle += 1
                yield f": heartbeat\n\n"
                await asyncio.sleep(1)
            db.expire_all()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/v1/battles/{battle_id}/communications")
def communications(battle_id: str, db: Session = Depends(get_db)):
    row = db.query(SessionRow).filter(SessionRow.battle_id == battle_id).first()
    if not row or not row.battle_json:
        raise HTTPException(404, "Battle not found")
    return {"entries": row.battle_json.get("communications", [])}


@app.get("/api/v1/battles/{battle_id}/debrief")
def debrief(battle_id: str, db: Session = Depends(get_db)):
    from .domain.enums import Side

    row = db.query(SessionRow).filter(SessionRow.battle_id == battle_id).first()
    if not row:
        raise HTTPException(404, "Battle not found")
    battle = app_services.load_battle_from_row(row)
    if not battle:
        raise HTTPException(404, "Battle not found")
    return {
        "battle_id": battle.battle_id,
        "status": battle.status.value,
        "result": battle.result,
        "round": battle.round,
        "seed": battle.seed,
        "friendly_remaining": len(battle.living_units(Side.FRIENDLY)),
        "opposition_remaining": len(battle.living_units(Side.OPPOSITION)),
        "event_count": len(battle.events),
        "friendly_vp": getattr(battle, "friendly_vp", 0),
        "opposition_vp": getattr(battle, "opposition_vp", 0),
        "vp_to_win": getattr(battle, "vp_to_win", 5),
        "objective": "temple_control",
    }


@app.put("/api/v1/battles/{battle_id}/directives")
def directives(battle_id: str, body: CommandEnvelope, db: Session = Depends(get_db)):
    row = db.query(SessionRow).filter(SessionRow.battle_id == battle_id).first()
    if not row:
        raise HTTPException(404, "Battle not found")
    battle = app_services.load_battle_from_row(row)
    text = body.payload.get("text", "")
    unit_id = body.payload.get("target_unit_id")
    order_id = body.payload.get("order_id")
    target_refs = body.payload.get("target_refs") or []
    try:
        return app_services.set_directives(
            db,
            row,
            battle,
            text,
            unit_id,
            command_id=body.command_id,
            order_id=order_id,
            target_refs=target_refs,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/v1/battles/{battle_id}/commander/actions")
def commander_actions(battle_id: str, body: CommandEnvelope, db: Session = Depends(get_db)):
    row = db.query(SessionRow).filter(SessionRow.battle_id == battle_id).first()
    if not row:
        raise HTTPException(404, "Battle not found")
    battle = app_services.load_battle_from_row(row)
    if body.expected_state_version is None:
        raise HTTPException(400, "expected_state_version required")
    try:
        return app_services.commander_action(
            db, row, battle, body.command_id, body.expected_state_version, body.payload["option_id"]
        )
    except app_services.ConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "STATE_VERSION_CONFLICT", "current_state_version": exc.current_version, "battle": exc.snapshot},
        ) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/v1/battles/{battle_id}/commander/ram")
def commander_ram(battle_id: str, body: CommandEnvelope, db: Session = Depends(get_db)):
    row = db.query(SessionRow).filter(SessionRow.battle_id == battle_id).first()
    if not row:
        raise HTTPException(404, "Battle not found")
    battle = app_services.load_battle_from_row(row)
    if body.expected_state_version is None:
        raise HTTPException(400, "expected_state_version required")
    try:
        ability_id = str(body.payload.get("ability_id") or "")
        target_unit_id = body.payload.get("target_unit_id")
        q = body.payload.get("q")
        r = body.payload.get("r")
        return app_services.commander_cast_ram(
            db,
            row,
            battle,
            body.command_id,
            body.expected_state_version,
            ability_id,
            target_unit_id=target_unit_id,
            q=int(q) if q is not None else None,
            r=int(r) if r is not None else None,
        )
    except app_services.ConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "STATE_VERSION_CONFLICT", "current_state_version": exc.current_version, "battle": exc.snapshot},
        ) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/v1/battles/{battle_id}/commander/move")
def commander_move(battle_id: str, body: CommandEnvelope, db: Session = Depends(get_db)):
    row = db.query(SessionRow).filter(SessionRow.battle_id == battle_id).first()
    if not row:
        raise HTTPException(404, "Battle not found")
    battle = app_services.load_battle_from_row(row)
    if body.expected_state_version is None:
        raise HTTPException(400, "expected_state_version required")
    try:
        q = int(body.payload["q"])
        r = int(body.payload["r"])
        return app_services.commander_move_to(db, row, battle, body.command_id, body.expected_state_version, q, r)
    except app_services.ConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "STATE_VERSION_CONFLICT", "current_state_version": exc.current_version, "battle": exc.snapshot},
        ) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/v1/battles/{battle_id}/commander/end-activation")
def commander_end(battle_id: str, body: CommandEnvelope, db: Session = Depends(get_db)):
    row = db.query(SessionRow).filter(SessionRow.battle_id == battle_id).first()
    if not row:
        raise HTTPException(404, "Battle not found")
    battle = app_services.load_battle_from_row(row)
    if body.expected_state_version is None:
        raise HTTPException(400, "expected_state_version required")
    try:
        return app_services.commander_end(db, row, battle, body.command_id, body.expected_state_version)
    except app_services.ConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "STATE_VERSION_CONFLICT", "current_state_version": exc.current_version, "battle": exc.snapshot},
        ) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _control_phase_http(battle_id: str, body: CommandEnvelope, db: Session, op: str):
    row = db.query(SessionRow).filter(SessionRow.battle_id == battle_id).first()
    if not row:
        raise HTTPException(404, "Battle not found")
    battle = app_services.load_battle_from_row(row)
    if body.expected_state_version is None:
        raise HTTPException(400, "expected_state_version required")
    try:
        if op == "allocate":
            drone_id = str(body.payload.get("drone_id") or "")
            return app_services.control_phase_allocate(
                db, row, battle, body.command_id, body.expected_state_version, drone_id
            )
        if op == "reclaim":
            drone_id = str(body.payload.get("drone_id") or "")
            return app_services.control_phase_reclaim(
                db, row, battle, body.command_id, body.expected_state_version, drone_id
            )
        return app_services.control_phase_complete(db, row, battle, body.command_id, body.expected_state_version)
    except app_services.ConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "STATE_VERSION_CONFLICT", "current_state_version": exc.current_version, "battle": exc.snapshot},
        ) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/v1/battles/{battle_id}/control-phase/allocate")
def control_phase_allocate(battle_id: str, body: CommandEnvelope, db: Session = Depends(get_db)):
    return _control_phase_http(battle_id, body, db, "allocate")


@app.post("/api/v1/battles/{battle_id}/control-phase/reclaim")
def control_phase_reclaim(battle_id: str, body: CommandEnvelope, db: Session = Depends(get_db)):
    return _control_phase_http(battle_id, body, db, "reclaim")


@app.post("/api/v1/battles/{battle_id}/control-phase/complete")
def control_phase_complete(battle_id: str, body: CommandEnvelope, db: Session = Depends(get_db)):
    return _control_phase_http(battle_id, body, db, "complete")


@app.post("/api/v1/battles/{battle_id}/resolve-next")
def resolve_next(battle_id: str, body: CommandEnvelope, db: Session = Depends(get_db)):
    row = db.query(SessionRow).filter(SessionRow.battle_id == battle_id).first()
    if not row:
        raise HTTPException(404, "Battle not found")
    battle = app_services.load_battle_from_row(row)
    if body.expected_state_version is None:
        raise HTTPException(400, "expected_state_version required")
    try:
        return app_services.resolve_next_agent(db, row, battle, body.command_id, body.expected_state_version)
    except app_services.ConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "STATE_VERSION_CONFLICT", "current_state_version": exc.current_version, "battle": exc.snapshot},
        ) from exc


@app.post("/api/v1/battles/{battle_id}/abort")
def abort_battle(battle_id: str, db: Session = Depends(get_db)):
    row = db.query(SessionRow).filter(SessionRow.battle_id == battle_id).first()
    if not row:
        raise HTTPException(404, "Battle not found")
    battle = app_services.load_battle_from_row(row)
    from .domain.enums import BattleStatus
    from .domain.events import DomainEvent

    battle.status = BattleStatus.ABORTED
    battle.result = "ABORTED"
    battle.append_events([DomainEvent(type="battle_aborted", payload={})])
    row.status = "ENDED"
    app_services.persist_battle(db, row, battle)
    return {"battle": app_services.battle_snapshot(battle, [])}


@app.post("/api/v1/sessions/{session_id}/feedback")
def feedback(session_id: str, body: CommandEnvelope, db: Session = Depends(get_db)):
    row = db.get(SessionRow, session_id)
    if not row:
        raise HTTPException(404, "Session not found")
    message = str(body.payload.get("message") or "").strip()
    if not message:
        raise HTTPException(400, "message required")
    battle_json = row.battle_json or {}
    consented = body.payload.get("context")
    if not isinstance(consented, dict):
        consented = {}
    db.add(
        FeedbackRow(
            session_id=session_id,
            message=message[:4000],
            context_json={
                "battle_id": row.battle_id,
                "status": battle_json.get("status") or battle_json.get("result"),
                "result": battle_json.get("result"),
                "round": battle_json.get("round"),
                "seed": battle_json.get("seed"),
                "friendly_vp": battle_json.get("friendly_vp"),
                "opposition_vp": battle_json.get("opposition_vp"),
                "map_id": battle_json.get("map_id"),
                "point_cap": battle_json.get("point_cap"),
                "consented_context": consented,
            },
        )
    )
    db.commit()
    structured_log(
        "feedback_submitted",
        session_id=session_id,
        battle_id=row.battle_id,
        message_len=len(message),
    )
    return {"ok": True, "session_id": session_id, "battle_id": row.battle_id}


@app.get("/api/v1/diagnostics/sessions")
def diagnostics_sessions(limit: int = 50, db: Session = Depends(get_db)):
    return {"sessions": diag.list_sessions(db, limit=min(limit, 200))}


@app.get("/api/v1/diagnostics/sessions/{session_id}")
def diagnostics_session(session_id: str, db: Session = Depends(get_db)):
    data = diag.session_diagnostics(db, session_id)
    if not data:
        raise HTTPException(404, "Session not found")
    return data


@app.get("/api/v1/diagnostics/battles/{battle_id}")
def diagnostics_battle(battle_id: str, db: Session = Depends(get_db)):
    data = diag.battle_diagnostics(db, battle_id)
    if not data:
        raise HTTPException(404, "Battle not found")
    return data


@app.post("/api/v1/dev/simulate")
def simulate(seed: int = 42):
    return app_services.run_headless_simulation(seed=seed)
