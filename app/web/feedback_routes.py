from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth import get_optional_player
from app.config import get_settings
from app.db import get_session
from app.models.feedback import Feedback
from app.models.match import Match, Player

router = APIRouter()
templates = Jinja2Templates(directory="app/web/templates")


@router.get("/feedback")
async def feedback_page(
    request: Request,
    player: Player | None = Depends(get_optional_player),
):
    return templates.TemplateResponse("feedback.html", {
        "request": request,
        "player": player,
        "sent": request.query_params.get("sent") == "1",
    })


@router.post("/feedback")
async def submit_feedback(
    request: Request,
    text: str = Form(...),
    rating: str = Form(""),
    player: Player | None = Depends(get_optional_player),
    db: Session = Depends(get_session),
):
    text = text.strip()[:2000]
    if not text:
        raise HTTPException(status_code=422, detail="Feedback text required.")

    rating_int = int(rating) if rating.strip() and rating.strip().isdigit() and 1 <= int(rating) <= 5 else None
    referer = request.headers.get("referer", "")

    entry = Feedback(
        id=str(uuid.uuid4()),
        text=text,
        rating=rating_int,
        username=player.username if player else None,
        page_url=referer[:512] if referer else None,
        created_at=datetime.utcnow(),
    )
    db.add(entry)
    db.commit()

    return RedirectResponse("/feedback?sent=1", status_code=303)


@router.get("/admin/users")
async def admin_users(
    request: Request,
    player: Player | None = Depends(get_optional_player),
    db: Session = Depends(get_session),
):
    settings = get_settings()
    if not settings.admin_username:
        raise HTTPException(status_code=404)
    if not player or player.username.lower() != settings.admin_username.lower():
        raise HTTPException(status_code=403, detail="Not authorised.")

    players = db.execute(
        select(Player).order_by(Player.created_at.desc())
    ).scalars().all()

    match_counts = {
        row[0]: row[1]
        for row in db.execute(
            select(Match.player_id, func.count(Match.id))
            .group_by(Match.player_id)
        ).all()
    }

    return templates.TemplateResponse("admin_users.html", {
        "request": request,
        "player": player,
        "players": players,
        "match_counts": match_counts,
        "total": len(players),
    })


@router.get("/admin/feedback")
async def admin_feedback(
    request: Request,
    format: str = "",
    player: Player | None = Depends(get_optional_player),
    db: Session = Depends(get_session),
):
    settings = get_settings()

    if not settings.admin_username:
        raise HTTPException(status_code=404)
    if not player or player.username.lower() != settings.admin_username.lower():
        raise HTTPException(status_code=403, detail="Not authorised.")

    entries = db.execute(
        select(Feedback).order_by(Feedback.created_at.desc())
    ).scalars().all()

    if format == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["created_at", "username", "rating", "page_url", "text"])
        for e in entries:
            writer.writerow([
                e.created_at.isoformat() if e.created_at else "",
                e.username or "anonymous",
                e.rating or "",
                e.page_url or "",
                e.text,
            ])
        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=feedback.csv"},
        )

    return templates.TemplateResponse("admin_feedback.html", {
        "request": request,
        "player": player,
        "entries": entries,
        "total": len(entries),
    })
