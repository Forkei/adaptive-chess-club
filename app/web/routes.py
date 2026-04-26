from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.auth import (
    COOKIE_MAX_AGE,
    PLAYER_COOKIE,
    EmailError,
    PasswordError,
    UsernameError,
    consume_password_reset_token,
    find_player_by_email,
    find_player_by_identifier,
    find_player_by_username,
    generate_guest_username,
    get_optional_player,
    hash_password,
    is_guest_username,
    issue_password_reset_token,
    needs_rehash,
    normalize_email,
    normalize_username,
    require_player,
    validate_email_address,
    validate_password,
    validate_username,
    verify_password,
)
from app.mail import Email, send as send_email
from app.characters.chat_service import (
    close_session as chat_close_session,
    get_or_create_session as chat_get_or_create_session,
    get_turns as chat_get_turns,
    handle_player_message as chat_handle_player_message,
)
from app.characters.openings import OPENINGS
from app.characters.rooms import theme_for_character
from app.characters.style import style_to_prompt_fragments
from app.db import get_session
from app.discovery import (
    character_leaderboard,
    hall_of_fame_for_character,
    list_live_matches,
    list_recent_matches,
    player_leaderboard,
)
from app.engine import EngineUnavailable, available_engines
from app.matches import service as match_service
from app.memory.crud import counts_by_scope, counts_by_type, list_for_character
from app.models.character import (
    Character,
    CharacterState,
    ContentRating,
    Visibility,
    rating_allowed,
    rating_level,
)
from app.models.match import Match, Player
from app.schemas.character import CharacterCreate, CharacterUpdate
from app.schemas.match import MoveRead, PlayerSettingsUpdate

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

router = APIRouter(tags=["web"])


def _room_for_match(character) -> "RoomTheme":
    """Return the room theme for a match page, swapping in ambient_track_game
    and ambient_volume_game when the character has a separate in-game track."""
    from app.characters.rooms import RoomTheme, theme_for_character
    room = theme_for_character(character)
    if room.ambient_track_game:
        room = replace(
            room,
            ambient_track=room.ambient_track_game,
            ambient_volume=room.ambient_volume_game,
        )
    return room


def _run_generation_bg(character_id: str) -> None:
    from app.characters.memory_generator import generate_and_store

    try:
        generate_and_store(character_id)
    except Exception:
        logger.exception("Memory generation failed for %s", character_id)


def _visible_filter(player: Player):
    """Build the WHERE clause for characters visible to this player."""
    max_idx = rating_level(player.max_content_rating)
    allowed = [
        r for r in (ContentRating.FAMILY, ContentRating.MATURE, ContentRating.UNRESTRICTED)
        if rating_level(r) <= max_idx
    ]
    return [
        Character.deleted_at.is_(None),
        Character.content_rating.in_(allowed),
        or_(Character.visibility == Visibility.PUBLIC, Character.owner_id == player.id),
    ]


# ------------------------------ Landing ------------------------------


@router.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    player: Player | None = Depends(get_optional_player),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    if player is None:
        return templates.TemplateResponse(
            request,
            "landing.html",
            {"player": None},
        )

    # Single-character mode: redirect logged-in users straight to Kenji's room.
    kenji = session.execute(
        select(Character).where(Character.preset_key == "kenji_sato")
    ).scalar_one_or_none()
    if kenji is not None:
        return RedirectResponse(url=f"/characters/{kenji.id}", status_code=302)

    # Fallback: Kenji not seeded yet (fresh DB before first startup completes).
    logger.warning("Kenji (preset_key='kenji_sato') not found — falling back to character grid")
    stmt = (
        select(Character)
        .where(*_visible_filter(player))
        .order_by(Character.is_preset.desc(), Character.created_at.desc())
    )
    chars = list(session.execute(stmt).scalars())

    owner_ids = {c.owner_id for c in chars if c.owner_id}
    owner_map: dict[str, str] = {}
    if owner_ids:
        owners = session.execute(
            select(Player).where(Player.id.in_(owner_ids))
        ).scalars()
        owner_map = {p.id: p.username for p in owners}

    return templates.TemplateResponse(
        request,
        "index.html",
        {"player": player, "characters": chars, "owner_map": owner_map},
    )


# ------------------------------ Discovery ------------------------------


def _kenji_character_id(session) -> str | None:
    """Return Kenji's character UUID, or None if not seeded yet."""
    kenji = session.execute(
        select(Character).where(Character.preset_key == "kenji_sato")
    ).scalar_one_or_none()
    return kenji.id if kenji is not None else None


@router.get("/leaderboard/characters", response_class=HTMLResponse)
def leaderboard_characters_page(
    request: Request,
    window: str = "all",
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    if window not in ("all", "30d", "7d"):
        window = "all"
    rows = character_leaderboard(
        session, viewer=player, window=window, character_id=_kenji_character_id(session)
    )
    return templates.TemplateResponse(
        request,
        "leaderboard_characters.html",
        {"player": player, "rows": rows, "window": window},
    )


@router.get("/leaderboard/players", response_class=HTMLResponse)
def leaderboard_players_page(
    request: Request,
    window: str = "all",
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    if window not in ("all", "30d", "7d"):
        window = "all"
    rows = player_leaderboard(
        session, viewer=player, window=window, character_id=_kenji_character_id(session)
    )
    return templates.TemplateResponse(
        request,
        "leaderboard_players.html",
        {"player": player, "rows": rows, "window": window},
    )


@router.get("/players/{username}", response_class=HTMLResponse)
def player_profile_page(
    request: Request,
    username: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    target = find_player_by_username(session, username)
    if target is None:
        raise HTTPException(status_code=404, detail="Player not found")

    # Recent matches by this player that are visible to the viewer.
    from app.models.match import Match as _Match, MatchStatus as _MS
    from app.discovery.queries import visible_character_filter as _vcf

    m_stmt = (
        select(_Match, Character)
        .join(Character, Character.id == _Match.character_id)
        .where(
            _Match.player_id == target.id,
            _Match.status.in_([_MS.COMPLETED, _MS.ABANDONED]),
            *_vcf(player),
        )
        .order_by(_Match.ended_at.desc().nulls_last())
        .limit(20)
    )
    recent_rows = list(session.execute(m_stmt).all())

    # Characters owned by this player, visible to the viewer.
    c_stmt = (
        select(Character)
        .where(
            Character.owner_id == target.id,
            *_visible_filter(player),
        )
        .order_by(Character.created_at.desc())
    )
    owned = list(session.execute(c_stmt).scalars())

    return templates.TemplateResponse(
        request,
        "player_profile.html",
        {
            "player": player,
            "target": target,
            "recent_matches": recent_rows,
            "owned_characters": owned,
            "owner_map": {target.id: target.username},
        },
    )


@router.get("/discovery", response_class=HTMLResponse)
def discovery(
    request: Request,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    kenji_id = _kenji_character_id(session)
    live = list_live_matches(session, viewer=player, limit=20, character_id=kenji_id)
    recent = list_recent_matches(session, viewer=player, limit=20, character_id=kenji_id)

    return templates.TemplateResponse(
        request,
        "discovery.html",
        {
            "player": player,
            "live_matches": live,
            "recent_matches": recent,
        },
    )


# ------------------------------ Auth ------------------------------


def _safe_next(raw: str) -> str:
    """Prevent open-redirects. Only accept in-app paths."""
    if not raw:
        return "/"
    if raw.startswith("/") and not raw.startswith("//"):
        return raw
    return "/"


def _is_https(request: Request) -> bool:
    # Respect the X-Forwarded-Proto header set by reverse proxies (HF Spaces, nginx).
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    return proto == "https"


def _set_login_cookie(response: RedirectResponse, player_id: str, *, https: bool = False) -> None:
    response.set_cookie(
        key=PLAYER_COOKIE,
        value=player_id,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=https,
    )


@router.get("/login", response_class=HTMLResponse)
def login_form(
    request: Request,
    next: str = "/",
    error: str | None = None,
    flash: str | None = None,
    identifier: str = "",
    player: Player | None = Depends(get_optional_player),
) -> HTMLResponse:
    prefill_identifier = identifier
    if not prefill_identifier and player is not None and is_guest_username(player.username):
        prefill_identifier = ""
    from app.characters.rooms import KENJI_ROOM
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "player": player,
            "next": next,
            "error": error,
            "flash": flash,
            "prefill_identifier": prefill_identifier,
            "room": KENJI_ROOM,
        },
    )


@router.post("/login")
def login_submit(
    request: Request,
    response: Response,
    identifier: str = Form(...),
    password: str = Form(""),
    next: str = Form("/"),
    player: Player | None = Depends(get_optional_player),
    session: Session = Depends(get_session),
):
    """Accept username OR email + password.

    Legacy (pre-4.0a) accounts and `guest_*` rows may have
    `password_hash=NULL`. For those, login proceeds with just the
    identifier (so existing dev accounts keep working); the UI nudges
    them to set a password via /settings.
    """
    identifier_raw = (identifier or "").strip()
    if not identifier_raw:
        return RedirectResponse(
            url=f"/login?next={next}&error=identifier_required",
            status_code=303,
        )

    existing = find_player_by_identifier(session, identifier_raw)
    if existing is None:
        # Don't distinguish "unknown user" from "wrong password" — same code.
        return RedirectResponse(
            url=f"/login?next={next}&error=bad_credentials&identifier={identifier_raw}",
            status_code=303,
        )

    if existing.password_hash:
        if not password or not verify_password(password, existing.password_hash):
            return RedirectResponse(
                url=f"/login?next={next}&error=bad_credentials&identifier={identifier_raw}",
                status_code=303,
            )
        # Upgrade hash params if argon2 defaults changed since signup.
        if needs_rehash(existing.password_hash):
            existing.password_hash = hash_password(password)
            session.commit()
    else:
        # Legacy / guest row — no password on file. Accept identifier-only.
        # Hint the user to set one.
        if password:
            # They provided a password, but the account has none. Don't reject
            # — treat it as a no-op and log them in; settings page nudges them.
            pass

    safe_next = _safe_next(next)
    separator = "&" if "?" in safe_next else "?"
    flash_key = "needs_password" if not existing.password_hash else "welcome_back"
    target = f"{safe_next}{separator}flash={flash_key}&u={existing.username}"
    redir = RedirectResponse(url=target, status_code=303)
    _set_login_cookie(redir, existing.id, https=_is_https(request))
    return redir


@router.get("/signup", response_class=HTMLResponse)
def signup_form(
    request: Request,
    next: str = "/",
    error: str | None = None,
    flash: str | None = None,
    player: Player | None = Depends(get_optional_player),
) -> HTMLResponse:
    # If a guest is logged in, pre-fill any currently-useful fields. We don't
    # prefill email because guests don't have one.
    from app.characters.rooms import KENJI_ROOM
    return templates.TemplateResponse(
        request,
        "signup.html",
        {
            "player": player,
            "next": next,
            "error": error,
            "flash": flash,
            "upgrade_guest": player is not None and is_guest_username(player.username),
            "room": KENJI_ROOM,
        },
    )


@router.post("/signup")
def signup_submit(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
    next: str = Form("/"),
    player: Player | None = Depends(get_optional_player),
    session: Session = Depends(get_session),
):
    username_n = normalize_username(username)
    email_raw = normalize_email(email)

    try:
        validate_username(username_n)
    except UsernameError as exc:
        return RedirectResponse(
            url=f"/signup?next={next}&error={exc.code}",
            status_code=303,
        )
    try:
        email_n = validate_email_address(email_raw)
    except EmailError as exc:
        return RedirectResponse(
            url=f"/signup?next={next}&error=email_{exc.code}",
            status_code=303,
        )
    try:
        validate_password(password)
    except PasswordError as exc:
        return RedirectResponse(
            url=f"/signup?next={next}&error=password_{exc.code}",
            status_code=303,
        )
    if password != password_confirm:
        return RedirectResponse(
            url=f"/signup?next={next}&error=password_mismatch",
            status_code=303,
        )

    # Uniqueness checks. Allow the current guest to keep their chosen
    # username when upgrading in-place.
    username_clash = find_player_by_username(session, username_n)
    if username_clash is not None and not (
        player is not None
        and player.id == username_clash.id
        and is_guest_username(player.username)
    ):
        return RedirectResponse(
            url=f"/signup?next={next}&error=username_taken",
            status_code=303,
        )
    email_clash = find_player_by_email(session, email_n)
    if email_clash is not None and not (
        player is not None and player.id == email_clash.id
    ):
        return RedirectResponse(
            url=f"/signup?next={next}&error=email_taken",
            status_code=303,
        )

    pwd_hash = hash_password(password)
    if player is not None and is_guest_username(player.username):
        # Upgrade the guest row in place — preserves matches, memories, Elo.
        player.username = username_n
        player.email = email_n
        player.password_hash = pwd_hash
        if player.display_name in ("Guest", "", None):
            player.display_name = username_n
        session.commit()
        session.refresh(player)
        final_player = player
        flash_key = "upgraded"
    else:
        final_player = Player(
            username=username_n,
            display_name=username_n,
            email=email_n,
            password_hash=pwd_hash,
        )
        session.add(final_player)
        session.commit()
        session.refresh(final_player)
        flash_key = "welcome"

    safe_next = _safe_next(next)
    separator = "&" if "?" in safe_next else "?"
    target = f"{safe_next}{separator}flash={flash_key}&u={final_player.username}"
    redir = RedirectResponse(url=target, status_code=303)
    _set_login_cookie(redir, final_player.id, https=_is_https(request))
    return redir


@router.get("/forgot-password", response_class=HTMLResponse)
def forgot_password_form(
    request: Request,
    error: str | None = None,
    flash: str | None = None,
    player: Player | None = Depends(get_optional_player),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "forgot_password.html",
        {"player": player, "error": error, "flash": flash},
    )


@router.post("/forgot-password")
def forgot_password_submit(
    request: Request,
    email: str = Form(...),
    session: Session = Depends(get_session),
):
    email_raw = normalize_email(email)
    # Try to validate but don't leak: any invalid input still shows the
    # generic success flash so attackers can't enumerate accounts.
    target = find_player_by_email(session, email_raw) if email_raw else None
    if target is not None:
        raw_token = issue_password_reset_token(session, target)
        session.commit()
        from app.config import get_settings as _gs

        reset_url = str(request.url_for("reset_password_form", token=raw_token))
        ttl = _gs().password_reset_token_ttl_minutes
        send_email(
            Email(
                to=target.email or email_raw,
                subject="Reset your Metropolis Chess Club password",
                body=(
                    f"Hello {target.username},\n\n"
                    f"To set a new password, visit:\n{reset_url}\n\n"
                    f"The link expires in {ttl} minutes.\n\n"
                    "If you didn't request this, you can ignore this message."
                ),
            )
        )
    return RedirectResponse(url="/forgot-password?flash=sent", status_code=303)


@router.get("/reset-password/{token}", response_class=HTMLResponse, name="reset_password_form")
def reset_password_form(
    request: Request,
    token: str,
    error: str | None = None,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "reset_password.html",
        {"token": token, "error": error},
    )


@router.post("/reset-password/{token}")
def reset_password_submit(
    request: Request,
    token: str,
    password: str = Form(...),
    password_confirm: str = Form(...),
    session: Session = Depends(get_session),
):
    try:
        validate_password(password)
    except PasswordError as exc:
        return RedirectResponse(
            url=f"/reset-password/{token}?error=password_{exc.code}",
            status_code=303,
        )
    if password != password_confirm:
        return RedirectResponse(
            url=f"/reset-password/{token}?error=password_mismatch",
            status_code=303,
        )

    player = consume_password_reset_token(session, token)
    if player is None:
        session.commit()  # persist any used_at markers
        return RedirectResponse(url="/login?error=reset_token_invalid", status_code=303)

    player.password_hash = hash_password(password)
    session.commit()

    redir = RedirectResponse(url="/?flash=password_reset", status_code=303)
    _set_login_cookie(redir, player.id, https=_is_https(request))
    return redir


@router.get("/logout")
def logout(response: Response):
    redir = RedirectResponse(url="/login", status_code=303)
    redir.delete_cookie(PLAYER_COOKIE)
    return redir


@router.get("/settings", response_class=HTMLResponse)
def settings_page(
    request: Request,
    player: Player = Depends(require_player),
    flash: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "player": player,
            "flash": flash,
            "error": error,
            "has_password": player.password_hash is not None,
            "has_email": player.email is not None,
        },
    )


@router.post("/settings")
def settings_submit(
    display_name: str = Form(...),
    max_content_rating: str = Form(...),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    try:
        rating = ContentRating(max_content_rating)
    except ValueError:
        return RedirectResponse(url="/settings?error=invalid_rating", status_code=303)
    display_name = display_name.strip()
    if not display_name:
        return RedirectResponse(url="/settings?error=empty_display_name", status_code=303)
    if len(display_name) > 80:
        return RedirectResponse(url="/settings?error=display_name_too_long", status_code=303)
    player.display_name = display_name
    player.max_content_rating = rating
    session.commit()
    return RedirectResponse(url="/settings?flash=saved", status_code=303)


@router.post("/settings/password")
def settings_change_password(
    current_password: str = Form(""),
    new_password: str = Form(...),
    new_password_confirm: str = Form(...),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    """Set-or-change the caller's password.

    - If `player.password_hash` is NULL (legacy / guest), `current_password`
      is ignored and the new password is simply set.
    - Otherwise `current_password` must verify against the stored hash.
    """
    if player.password_hash:
        if not verify_password(current_password, player.password_hash):
            return RedirectResponse(
                url="/settings?error=wrong_current_password", status_code=303
            )
    try:
        validate_password(new_password)
    except PasswordError as exc:
        return RedirectResponse(
            url=f"/settings?error=password_{exc.code}", status_code=303
        )
    if new_password != new_password_confirm:
        return RedirectResponse(
            url="/settings?error=password_mismatch", status_code=303
        )
    player.password_hash = hash_password(new_password)
    session.commit()
    return RedirectResponse(url="/settings?flash=password_saved", status_code=303)


@router.post("/settings/email")
def settings_change_email(
    new_email: str = Form(...),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    raw = normalize_email(new_email)
    try:
        addr = validate_email_address(raw)
    except EmailError as exc:
        return RedirectResponse(
            url=f"/settings?error=email_{exc.code}", status_code=303
        )
    clash = find_player_by_email(session, addr)
    if clash is not None and clash.id != player.id:
        return RedirectResponse(url="/settings?error=email_taken", status_code=303)
    player.email = addr
    # Email verification not implemented in 4.0a — treat as unverified.
    player.email_verified_at = None
    session.commit()
    return RedirectResponse(url="/settings?flash=email_saved", status_code=303)


# ------------------------------ Characters ------------------------------


@router.get("/characters/new", response_class=HTMLResponse)
def new_character_form(
    request: Request,
    player: Player = Depends(require_player),
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "new.html",
        {"player": player, "openings": OPENINGS},
    )


@router.post("/characters/new")
def create_character_html(
    request: Request,
    background: BackgroundTasks,
    name: str = Form(...),
    short_description: str = Form(""),
    backstory: str = Form(""),
    avatar_emoji: str = Form("♟️"),
    aggression: int = Form(5),
    risk_tolerance: int = Form(5),
    patience: int = Form(5),
    trash_talk: int = Form(5),
    target_elo: int = Form(1400),
    adaptive: str = Form(""),
    opening_preferences: list[str] = Form(default=[]),
    voice_descriptor: str = Form(""),
    quirks: str = Form(""),
    visibility: str = Form("public"),
    content_rating: str = Form("family"),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> RedirectResponse:
    try:
        vis = Visibility(visibility)
    except ValueError:
        vis = Visibility.PUBLIC
    try:
        rating = ContentRating(content_rating)
    except ValueError:
        rating = ContentRating.FAMILY

    payload = CharacterCreate(
        name=name,
        short_description=short_description,
        backstory=backstory,
        avatar_emoji=avatar_emoji or "♟️",
        aggression=aggression,
        risk_tolerance=risk_tolerance,
        patience=patience,
        trash_talk=trash_talk,
        target_elo=target_elo,
        adaptive=bool(adaptive),
        opening_preferences=opening_preferences,
        voice_descriptor=voice_descriptor,
        quirks=quirks,
        visibility=vis,
        content_rating=rating,
    )
    character = Character(
        name=payload.name,
        short_description=payload.short_description,
        backstory=payload.backstory,
        avatar_emoji=payload.avatar_emoji,
        aggression=payload.aggression,
        risk_tolerance=payload.risk_tolerance,
        patience=payload.patience,
        trash_talk=payload.trash_talk,
        target_elo=payload.target_elo,
        current_elo=payload.target_elo,
        floor_elo=payload.target_elo,
        max_elo=payload.max_elo if payload.max_elo is not None else payload.target_elo + 400,
        adaptive=payload.adaptive,
        opening_preferences=list(payload.opening_preferences),
        voice_descriptor=payload.voice_descriptor,
        quirks=payload.quirks,
        visibility=payload.visibility,
        content_rating=payload.content_rating,
        owner_id=player.id,
        state=CharacterState.GENERATING_MEMORIES,
        memory_generation_started_at=datetime.utcnow(),
        is_preset=False,
    )
    session.add(character)
    session.commit()
    session.refresh(character)
    background.add_task(_run_generation_bg, character.id)

    return RedirectResponse(url=f"/characters/{character.id}", status_code=303)


@router.get("/characters/{character_id}", response_class=HTMLResponse)
def character_detail(
    request: Request,
    character_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        # Stale bookmark from an old ephemeral DB — redirect home instead of 404.
        return RedirectResponse(url="/", status_code=302)
    if character.visibility == Visibility.PRIVATE and character.owner_id != player.id:
        raise HTTPException(status_code=404, detail="Character not found")
    if not rating_allowed(character.content_rating, player.max_content_rating):
        # Spec: "show as 'hidden by your content preference' if accessed directly"
        return templates.TemplateResponse(
            request,
            "rating_hidden.html",
            {"player": player, "character_rating": character.content_rating.value},
            status_code=403,
        )

    scope_counts = counts_by_scope(session, character_id=character_id)
    type_counts = counts_by_type(session, character_id=character_id)
    fragments = style_to_prompt_fragments(character)

    samples_by_type: dict[str, list] = {}
    for type_value in type_counts:
        from app.models.memory import MemoryType as MT

        try:
            mt = MT(type_value)
        except ValueError:
            continue
        rows, _ = list_for_character(
            session, character_id=character_id, type_=mt, offset=0, limit=3
        )
        samples_by_type[type_value] = rows

    owner_username = None
    if character.owner_id:
        owner = session.get(Player, character.owner_id)
        owner_username = owner.username if owner else None

    hof = hall_of_fame_for_character(session, character_id=character_id)

    return templates.TemplateResponse(
        request,
        "detail.html",
        {
            "player": player,
            "character": character,
            "scope_counts": scope_counts,
            "type_counts": type_counts,
            "total_memories": sum(scope_counts.values()),
            "fragments": fragments,
            "samples_by_type": samples_by_type,
            "is_generating": character.state == CharacterState.GENERATING_MEMORIES,
            "is_failed": character.state == CharacterState.GENERATION_FAILED,
            "owner_username": owner_username,
            "is_owner": character.owner_id == player.id,
            "hall_of_fame": hof,
            "room": theme_for_character(character),
        },
    )


# ------------------------------ Pre-match chat (Phase 4.2.5) ------------------------------


@router.get("/characters/{character_id}/chat/history")
def character_chat_history(
    character_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    """Return every turn of the current active session. Client calls this
    on room entry so a page refresh doesn't blank the conversation.
    """
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    # Enforce the same visibility/rating gate as the detail page.
    if character.visibility == Visibility.PRIVATE and character.owner_id != player.id:
        raise HTTPException(status_code=404, detail="Character not found")
    if not rating_allowed(character.content_rating, player.max_content_rating):
        raise HTTPException(status_code=403, detail="Content rating exceeds your preference.")

    chat_session = chat_get_or_create_session(session, character=character, player=player)

    # Phase 4.4d — character opens first (sometimes). Fires only on a
    # fresh, empty session; probabilistic so it doesn't feel scripted.
    # Sync Soul call is already tolerated on the POST path; the GET adds
    # at most one Soul round-trip on first room entry.
    try:
        from app.characters.chat_service import maybe_character_greets

        maybe_character_greets(session, chat_session, character, player)
    except Exception:
        logger.exception("[chat] greeting hook failed (non-fatal)")

    turns = chat_get_turns(session, chat_session)
    return {
        "session_id": chat_session.id,
        "status": chat_session.status.value,
        "handed_off_match_id": chat_session.handed_off_match_id,
        "turns": [
            {
                "id": t.id,
                "turn_number": t.turn_number,
                "role": t.role.value,
                "text": t.text,
                "emotion": t.emotion,
                "emotion_intensity": t.emotion_intensity,
                "game_action": t.game_action,
                "created_at": t.created_at.isoformat() + "Z",
            }
            for t in turns
        ],
    }


@router.post("/characters/{character_id}/chat")
async def character_chat_post(
    character_id: str,
    text: str = Form(...),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if character.visibility == Visibility.PRIVATE and character.owner_id != player.id:
        raise HTTPException(status_code=404, detail="Character not found")
    if not rating_allowed(character.content_rating, player.max_content_rating):
        raise HTTPException(status_code=403, detail="Content rating exceeds your preference.")
    if character.state != CharacterState.READY:
        raise HTTPException(status_code=409, detail="Character is still preparing its memories.")

    logger.info(
        "HTTP chat path hit for character=%s — client should use /room socket namespace",
        character_id,
    )

    chat_session = chat_get_or_create_session(session, character=character, player=player)

    try:
        result = chat_handle_player_message(
            session,
            chat_session=chat_session,
            character=character,
            player=player,
            text=text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Opening move intentionally NOT run here. The /play socket connect handler
    # fires it as a background task when move_count==0 and character is white.
    # Clients using the HTTP fallback will see the engine's move arrive once they
    # connect to the /play namespace.

    ct = result.character_turn
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=202,
        content={
            "session_id": chat_session.id,
            "player_turn": {
                "id": result.player_turn.id,
                "turn_number": result.player_turn.turn_number,
                "text": result.player_turn.text,
            },
            "character_turn": {
                "id": ct.id,
                "turn_number": ct.turn_number,
                "text": ct.text,
                "emotion": ct.emotion,
                "emotion_intensity": ct.emotion_intensity,
                "game_action": ct.game_action,
            },
            "surfaced_memories": [
                {
                    "memory_id": m.memory_id,
                    "narrative_text": m.narrative_text,
                    "retrieval_reason": m.retrieval_reason,
                    "from_cache": m.from_cache,
                }
                for m in result.surfaced_memories
            ],
            "handed_off_match_id": result.handed_off_match_id,
            "redirect_url": (
                f"/matches/{result.handed_off_match_id}" if result.handed_off_match_id else None
            ),
        },
    )


@router.post("/characters/{character_id}/chat/leave")
def character_chat_leave(
    character_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
):
    character = session.get(Character, character_id)
    if character is None:
        return {"closed": False}
    chat_session = chat_get_or_create_session(session, character=character, player=player)
    chat_close_session(session, chat_session)
    return {"closed": True}


@router.get("/characters/{character_id}/edit", response_class=HTMLResponse)
def edit_character_form(
    request: Request,
    character_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if character.is_preset:
        raise HTTPException(status_code=403, detail="Presets cannot be edited. Clone first.")
    if character.owner_id != player.id:
        raise HTTPException(status_code=403, detail="Not your character.")

    return templates.TemplateResponse(
        request,
        "edit.html",
        {"player": player, "character": character, "openings": OPENINGS},
    )


@router.post("/characters/{character_id}/edit")
def edit_character_submit(
    character_id: str,
    short_description: str = Form(""),
    backstory: str = Form(""),
    avatar_emoji: str = Form("♟️"),
    aggression: int = Form(5),
    risk_tolerance: int = Form(5),
    patience: int = Form(5),
    trash_talk: int = Form(5),
    voice_descriptor: str = Form(""),
    quirks: str = Form(""),
    opening_preferences: list[str] = Form(default=[]),
    visibility: str = Form("public"),
    content_rating: str = Form("family"),
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> RedirectResponse:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if character.is_preset:
        raise HTTPException(status_code=403, detail="Presets cannot be edited. Clone first.")
    if character.owner_id != player.id:
        raise HTTPException(status_code=403, detail="Not your character.")

    try:
        vis = Visibility(visibility)
    except ValueError:
        vis = character.visibility
    try:
        rating = ContentRating(content_rating)
    except ValueError:
        rating = character.content_rating

    update = CharacterUpdate(
        short_description=short_description,
        backstory=backstory,
        avatar_emoji=avatar_emoji or "♟️",
        aggression=aggression,
        risk_tolerance=risk_tolerance,
        patience=patience,
        trash_talk=trash_talk,
        voice_descriptor=voice_descriptor,
        quirks=quirks,
        opening_preferences=opening_preferences,
        visibility=vis,
        content_rating=rating,
    )
    for field, value in update.model_dump(exclude_unset=True, exclude_none=True).items():
        setattr(character, field, value)
    character.updated_at = datetime.utcnow()
    session.commit()
    return RedirectResponse(url=f"/characters/{character_id}", status_code=303)


@router.post("/characters/{character_id}/delete")
def delete_character_html(
    character_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> RedirectResponse:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if character.is_preset:
        raise HTTPException(status_code=403, detail="Preset characters cannot be deleted")
    if character.owner_id != player.id:
        raise HTTPException(status_code=403, detail="Not your character.")
    character.deleted_at = datetime.utcnow()
    session.commit()
    return RedirectResponse(url="/", status_code=303)


@router.post("/characters/{character_id}/clone")
def clone_character_html(
    character_id: str,
    background: BackgroundTasks,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> RedirectResponse:
    source = session.get(Character, character_id)
    if source is None or source.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if source.visibility == Visibility.PRIVATE and source.owner_id != player.id:
        raise HTTPException(status_code=404, detail="Character not found")
    if not rating_allowed(source.content_rating, player.max_content_rating):
        raise HTTPException(status_code=403, detail="Content rating exceeds your preference.")

    clone = Character(
        name=f"{source.name} (clone)",
        short_description=source.short_description,
        backstory=source.backstory,
        avatar_emoji=source.avatar_emoji,
        aggression=source.aggression,
        risk_tolerance=source.risk_tolerance,
        patience=source.patience,
        trash_talk=source.trash_talk,
        target_elo=source.target_elo,
        current_elo=source.target_elo,
        floor_elo=source.target_elo,
        max_elo=source.max_elo,
        adaptive=source.adaptive,
        opening_preferences=list(source.opening_preferences or []),
        voice_descriptor=source.voice_descriptor,
        quirks=source.quirks,
        visibility=Visibility.PUBLIC,
        content_rating=source.content_rating,
        owner_id=player.id,
        is_preset=False,
        preset_key=None,
        state=CharacterState.GENERATING_MEMORIES,
        memory_generation_started_at=datetime.utcnow(),
    )
    session.add(clone)
    session.commit()
    session.refresh(clone)
    background.add_task(_run_generation_bg, clone.id)
    return RedirectResponse(url=f"/characters/{clone.id}", status_code=303)


@router.post("/characters/{character_id}/regenerate")
def regenerate_memories_html(
    character_id: str,
    background: BackgroundTasks,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> RedirectResponse:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if character.is_preset:
        raise HTTPException(status_code=403, detail="Presets cannot be regenerated.")
    if character.owner_id != player.id:
        raise HTTPException(status_code=403, detail="Not your character.")

    from app.models.memory import Memory, MemoryScope as MS

    session.query(Memory).filter(
        Memory.character_id == character_id,
        Memory.scope.in_([MS.CHARACTER_LORE, MS.CROSS_PLAYER]),
    ).delete(synchronize_session=False)
    character.state = CharacterState.GENERATING_MEMORIES
    character.memory_generation_started_at = datetime.utcnow()
    character.memory_generation_error = None
    session.commit()
    background.add_task(_run_generation_bg, character_id)
    return RedirectResponse(url=f"/characters/{character_id}", status_code=303)


# ------------------------------ Matches ------------------------------


@router.post("/play/{character_id}")
async def start_match_html(
    character_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> RedirectResponse:
    character = session.get(Character, character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Character not found")
    if character.visibility == Visibility.PRIVATE and character.owner_id != player.id:
        raise HTTPException(status_code=404, detail="Character not found")
    if not rating_allowed(character.content_rating, player.max_content_rating):
        raise HTTPException(status_code=403, detail="Content rating exceeds your preference.")

    try:
        match = match_service.create_match(
            session,
            character_id=character.id,
            player_id=player.id,
            player_color="random",
        )
        session.commit()
        await match_service.start_match_play(session, match)
        session.commit()
    except EngineUnavailable as exc:
        session.rollback()
        logger.exception("Engine unavailable starting match for %s", character_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except match_service.MatchError as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RedirectResponse(url=f"/matches/{match.id}", status_code=303)


@router.get("/matches/{match_id}", response_class=HTMLResponse)
def match_page(
    request: Request,
    match_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    match = session.get(Match, match_id)
    if match is None or match.player_id != player.id:
        raise HTTPException(status_code=404, detail="Match not found")
    character = session.get(Character, match.character_id)

    moves = [
        MoveRead.model_validate(m).model_dump(mode="json")
        for m in sorted(match.moves, key=lambda m: m.move_number)
    ]

    engines = available_engines()
    real_engines = [e for e in engines if e != "mock"]

    # Pre-match chat continuity: when the match was handed off from a
    # character chat session, pull the turns so play.html can seed the
    # transcript with the conversation that led up to the game.
    pre_match_chat: list[dict] = []
    pre_session_id = (match.extra_state or {}).get("pre_match_chat_session_id")
    if pre_session_id:
        from app.models.chat import CharacterChatTurn, ChatTurnRole

        rows = session.execute(
            select(CharacterChatTurn)
            .where(CharacterChatTurn.session_id == pre_session_id)
            .order_by(CharacterChatTurn.turn_number.asc())
        ).scalars().all()
        for t in rows:
            pre_match_chat.append({
                "role": "player" if t.role == ChatTurnRole.PLAYER else "character",
                "text": t.text,
                "emotion": t.emotion,
                "emotion_intensity": t.emotion_intensity,
            })

    return templates.TemplateResponse(
        request,
        "play.html",
        {
            "player": player,
            "match": match,
            "character": character,
            "moves_json": moves,
            "engines_available": engines,
            "has_real_engine": bool(real_engines),
            "room": _room_for_match(character),
            "pre_match_chat": pre_match_chat,
        },
    )


@router.get("/matches/{match_id}/watch", response_class=HTMLResponse, response_model=None)
def match_watch_page(
    request: Request,
    match_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> HTMLResponse | RedirectResponse:
    match = session.get(Match, match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")
    # Participants don't spectate — redirect them to their own play page.
    if match.player_id == player.id:
        return RedirectResponse(url=f"/matches/{match_id}", status_code=303)

    character = session.get(Character, match.character_id)
    if character is None or character.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Match not found")
    if character.visibility == Visibility.PRIVATE and character.owner_id != player.id:
        raise HTTPException(status_code=404, detail="Match not found")
    if not rating_allowed(character.content_rating, player.max_content_rating):
        raise HTTPException(status_code=403, detail="Content rating exceeds your preference.")

    # Abandoned matches: sparse and not interesting to a non-participant.
    if match.status.value == "abandoned":
        raise HTTPException(status_code=404, detail="This match ended abruptly — nothing to watch.")

    # Completed matches: allow — the watch view shows final state + full move list.
    match_owner = session.get(Player, match.player_id)
    moves = [
        MoveRead.model_validate(m).model_dump(mode="json")
        for m in sorted(match.moves, key=lambda m: m.move_number)
    ]

    return templates.TemplateResponse(
        request,
        "watch.html",
        {
            "player": player,
            "match": match,
            "character": character,
            "match_owner": match_owner,
            "moves_json": moves,
            "room": _room_for_match(character),
        },
    )


@router.get("/matches/{match_id}/summary", response_class=HTMLResponse)
def match_summary_page(
    request: Request,
    match_id: str,
    player: Player = Depends(require_player),
    session: Session = Depends(get_session),
) -> HTMLResponse:
    from app.matches.service import player_outcome
    from app.models.match import MatchAnalysis, OpponentProfile
    from app.models.memory import Memory
    from app.post_match.elo_apply import compute_elo_delta

    match = session.get(Match, match_id)
    if match is None or match.player_id != player.id:
        raise HTTPException(status_code=404, detail="Match not found")
    character = session.get(Character, match.character_id)

    analysis = session.execute(
        select(MatchAnalysis).where(MatchAnalysis.match_id == match_id)
    ).scalar_one_or_none()

    generated_memories: list = []
    if analysis and analysis.generated_memory_ids:
        generated_memories = list(
            session.execute(
                select(Memory).where(Memory.id.in_(list(analysis.generated_memory_ids)))
            ).scalars()
        )

    profile = session.execute(
        select(OpponentProfile).where(
            OpponentProfile.character_id == match.character_id,
            OpponentProfile.player_id == match.player_id,
        )
    ).scalar_one_or_none()

    elo_breakdown = None
    if analysis and analysis.elo_delta_raw is not None:
        moves = (analysis.engine_analysis or {}).get("moves") or []
        comp = compute_elo_delta(
            match=match,
            analysis_moves=moves,
            character_elo=match.character_elo_at_start,
            player_elo=match.player_elo_at_start or player.elo,
        )
        elo_breakdown = {
            "outcome": round(comp.outcome_delta, 1),
            "move_quality": round(comp.move_quality_delta, 1),
            "raw": round(comp.elo_delta_raw, 1),
            "short_halved": comp.short_match_halved,
            "rage_quit": comp.rage_quit_skipped_quality,
            "expected": round(comp.expected_score, 3),
            "actual": round(comp.actual_score, 2),
            "k": comp.k_factor,
            "player_raw": round(comp.player_elo_delta_raw, 1),
            "player_expected": round(comp.player_expected_score, 3),
            "player_actual": round(comp.player_actual_score, 2),
            "player_k": comp.player_k_factor,
        }

    return templates.TemplateResponse(
        request,
        "summary.html",
        {
            "player": player,
            "match": match,
            "character": character,
            "player_outcome": player_outcome(match),
            "char_color": "black" if match.player_color.value == "white" else "white",
            "elo_before": match.character_elo_at_start,
            "elo_after": match.character_elo_at_end or match.character_elo_at_start,
            "elo_delta_applied": analysis.elo_delta_applied if analysis else None,
            "floor_raised": bool(analysis.floor_raised) if analysis else False,
            "player_elo_before": match.player_elo_at_start,
            "player_elo_after": match.player_elo_at_end or match.player_elo_at_start,
            "player_elo_delta_applied": analysis.player_elo_delta_applied if analysis else None,
            "player_floor_raised": bool(analysis.player_floor_raised) if analysis else False,
            "elo_breakdown": elo_breakdown,
            "critical_moments": list(analysis.critical_moments or []) if analysis else [],
            "generated_memories": generated_memories,
            "narrative_summary": profile.narrative_summary if profile else None,
            "analysis_status": analysis.status.value if analysis else "none",
        },
    )
