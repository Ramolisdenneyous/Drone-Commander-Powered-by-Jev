"""Troubleshooting diagnostics for sessions, battles, agents, and request logs."""

from __future__ import annotations

from sqlalchemy import desc
from sqlalchemy.orm import Session

from ..persistence.models import (
    AgentRunRow,
    DomainEventRow,
    FeedbackRow,
    LLMArtifactRow,
    OutboxRow,
    RequestLogRow,
    SessionRow,
)


def list_sessions(db: Session, limit: int = 50) -> list[dict]:
    rows = db.query(SessionRow).order_by(desc(SessionRow.updated_at)).limit(limit).all()
    return [
        {
            "session_id": r.session_id,
            "status": r.status,
            "generation": r.generation,
            "state_version": r.state_version,
            "battle_id": r.battle_id,
            "content_version": r.content_version,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            "last_trace_id": r.last_trace_id,
            "battle_status": (r.battle_json or {}).get("status"),
            "battle_round": (r.battle_json or {}).get("round"),
            "battle_result": (r.battle_json or {}).get("result"),
        }
        for r in rows
    ]


def session_diagnostics(db: Session, session_id: str) -> dict | None:
    row = db.get(SessionRow, session_id)
    if not row:
        return None
    battle_id = row.battle_id
    req_q = db.query(RequestLogRow).filter(RequestLogRow.session_id == session_id)
    if battle_id:
        from sqlalchemy import or_

        req_q = db.query(RequestLogRow).filter(
            or_(RequestLogRow.session_id == session_id, RequestLogRow.battle_id == battle_id)
        )
    requests = req_q.order_by(desc(RequestLogRow.id)).limit(100).all()
    agent_runs = []
    artifacts = []
    events = []
    outbox = []
    if battle_id:
        agent_runs = (
            db.query(AgentRunRow)
            .filter(AgentRunRow.battle_id == battle_id)
            .order_by(desc(AgentRunRow.id))
            .limit(200)
            .all()
        )
        artifacts = (
            db.query(LLMArtifactRow)
            .filter(LLMArtifactRow.battle_id == battle_id)
            .order_by(desc(LLMArtifactRow.id))
            .limit(200)
            .all()
        )
        events = (
            db.query(DomainEventRow)
            .filter(DomainEventRow.battle_id == battle_id)
            .order_by(DomainEventRow.sequence.asc())
            .all()
        )
        outbox = (
            db.query(OutboxRow)
            .filter(OutboxRow.battle_id == battle_id)
            .order_by(desc(OutboxRow.id))
            .limit(100)
            .all()
        )

    fallback_count = sum(1 for r in agent_runs if r.fallback_used)
    feedback_rows = (
        db.query(FeedbackRow)
        .filter(FeedbackRow.session_id == session_id)
        .order_by(desc(FeedbackRow.id))
        .all()
    )
    return {
        "session": {
            "session_id": row.session_id,
            "status": row.status,
            "generation": row.generation,
            "state_version": row.state_version,
            "content_version": row.content_version,
            "battle_id": row.battle_id,
            "prep": row.prep_json,
            "last_trace_id": row.last_trace_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        },
        "battle_summary": {
            "status": (row.battle_json or {}).get("status"),
            "result": (row.battle_json or {}).get("result"),
            "round": (row.battle_json or {}).get("round"),
            "seed": (row.battle_json or {}).get("seed"),
            "active_actor_id": (row.battle_json or {}).get("active_actor_id"),
            "event_sequence": (row.battle_json or {}).get("event_sequence"),
            "state_version": (row.battle_json or {}).get("state_version"),
        },
        "stats": {
            "request_count": len(requests),
            "event_count": len(events),
            "agent_run_count": len(agent_runs),
            "fallback_count": fallback_count,
            "artifact_count": len(artifacts),
            "outbox_count": len(outbox),
            "feedback_count": len(feedback_rows),
        },
        "feedback": [
            {
                "id": f.id,
                "message": f.message,
                "context": f.context_json,
                "created_at": f.created_at.isoformat() if f.created_at else None,
            }
            for f in feedback_rows
        ],
        "recent_requests": [_request_dict(r) for r in requests[:40]],
        "recent_agent_runs": [_agent_dict(r) for r in agent_runs[:40]],
        "recent_artifacts": [_artifact_dict(a) for a in artifacts[:40]],
        "recent_event_types": [
            {"sequence": e.sequence, "type": e.event_type, "state_version": e.state_version}
            for e in events[-40:]
        ],
    }


def battle_diagnostics(db: Session, battle_id: str) -> dict | None:
    row = db.query(SessionRow).filter(SessionRow.battle_id == battle_id).first()
    if not row:
        return None
    data = session_diagnostics(db, row.session_id)
    if not data:
        return None
    runs = (
        db.query(AgentRunRow)
        .filter(AgentRunRow.battle_id == battle_id)
        .order_by(AgentRunRow.id.asc())
        .all()
    )
    events = (
        db.query(DomainEventRow)
        .filter(DomainEventRow.battle_id == battle_id)
        .order_by(DomainEventRow.sequence.asc())
        .all()
    )
    data["agent_runs"] = [_agent_dict(r) for r in runs]
    data["events_tail"] = [
        {
            "sequence": e.sequence,
            "type": e.event_type,
            "state_version": e.state_version,
            "actor_id": (e.envelope_json or {}).get("actor_id"),
            "payload_keys": list(((e.envelope_json or {}).get("payload") or {}).keys())[:12],
        }
        for e in events[-80:]
    ]
    return data


def _request_dict(r: RequestLogRow) -> dict:
    return {
        "trace_id": r.trace_id,
        "method": r.method,
        "path": r.path,
        "status_code": r.status_code,
        "duration_ms": r.duration_ms,
        "operation": r.operation,
        "error_code": r.error_code,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def _agent_dict(r: AgentRunRow) -> dict:
    artifact = r.artifact_json or {}
    return {
        "id": r.id,
        "activation_id": r.activation_id,
        "actor_id": r.actor_id,
        "round": r.round,
        "side": r.side,
        "definition_id": r.definition_id,
        "status": r.status,
        "selected_option_id": r.selected_option_id,
        "selected_subroutine": r.selected_subroutine,
        "fallback_used": r.fallback_used,
        "fallback_reason": r.fallback_reason,
        "provider": r.provider,
        "model": r.model,
        "confidence": artifact.get("confidence"),
        "margin": artifact.get("margin"),
        "provider_latency_ms": artifact.get("provider_latency_ms"),
        "shadow": bool(artifact.get("shadow")),
        "shadow_agreement": artifact.get("agreement"),
        "jev_option_id": artifact.get("jev_option_id"),
        "offered_option_count": r.offered_option_count,
        "issued_state_version": r.issued_state_version,
        "committed_state_version": r.committed_state_version,
        "duration_ms": r.duration_ms,
        "trace_id": r.trace_id,
        "artifact_json": r.artifact_json,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def _artifact_dict(a: LLMArtifactRow) -> dict:
    return {
        "id": a.id,
        "purpose": a.purpose,
        "provider": a.provider,
        "model": a.model,
        "prompt_hash": a.prompt_hash,
        "response_hash": a.response_hash,
        "token_usage": a.token_usage_json,
        "summary": a.summary_json,
        "success": a.success,
        "error": a.error,
        "duration_ms": a.duration_ms,
        "trace_id": a.trace_id,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }
