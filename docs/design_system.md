# Metropolis design system (Phase 3c polish pass)

A small, intentional vocabulary — not a framework. Tokens live in `app/web/templates/base.html`;
Tailwind (CDN) supplies the utility classes around them. No build step.

## Aesthetic direction

**"A discreet old-world chess café, crossed with a precise digital instrument."**
Brass and ink and felt and parchment, under warm incandescent light. Dark theme with warm undertones (not cool neutrals). Serifs for voice, mono for numbers, sans for body. Restraint beats density.

- Character cards feel like **member dossiers or trading cards**, not SaaS tiles.
- The board is the **hero** on the play page — framed in brass.
- Leaderboards are treated like **a club's posted standings**.
- The post-match summary is the **payoff screen** — half dossier, half quotation.

## Color tokens

Defined as CSS custom properties on `:root`. Reference via `var(--mp-…)`.

| Token                    | Value       | Use                                                      |
| ------------------------ | ----------- | -------------------------------------------------------- |
| `--mp-bg`                | `#0F1412`   | Page background (felt in shadow)                         |
| `--mp-surface-1`         | `#151B19`   | Primary panel                                            |
| `--mp-surface-2`         | `#1C2320`   | Raised card                                              |
| `--mp-surface-3`         | `#232A27`   | Hover lift                                               |
| `--mp-hairline`          | `#2E3430`   | Subtle border                                            |
| `--mp-hairline-2`        | `#3A4038`   | Stronger border                                          |
| `--mp-ink`               | `#EDE4CE`   | Primary text (parchment)                                 |
| `--mp-ink-muted`         | `#B5AD96`   | Secondary text                                           |
| `--mp-ink-faint`         | `#7F7966`   | Captions, metadata                                       |
| `--mp-ink-ghost`         | `#4C483E`   | Placeholders, dividers-in-text                           |
| `--mp-brass`             | `#C9A66B`   | Primary accent — links, highlights, CTA                  |
| `--mp-brass-bright`      | `#E0C38F`   | Emphasis / live-state / hover                            |
| `--mp-brass-dim`         | `#8E7649`   | Borders under brass elements                             |
| `--mp-felt`              | `#2F6B5C`   | In-progress / success                                    |
| `--mp-felt-bright`       | `#4FA48D`   | Live indicators                                          |
| `--mp-oxblood`           | `#A8423F`   | Resign / danger                                          |
| `--mp-ink-blue`          | `#5879A3`   | Spectator / secondary informational                      |
| `--mp-ink-blue-alt`      | `#8DA4C3`   | Spectator emphasis                                       |
| `--mp-rating-*`          | (see base) | Family / mature / unrestricted chips                     |

## Typography

| Role    | Font                  | Notes                                                                 |
| ------- | --------------------- | --------------------------------------------------------------------- |
| Display | **Fraunces**          | Optical sizing + SOFT/WONK variable axes for subtle character. Use `.mp-display` (standard), `.mp-italic` (quotations), `.mp-display-tight` (headings). |
| Body    | **IBM Plex Sans**     | Default on `<body>`. Weights 300/400/500/600 available.               |
| Mono    | **IBM Plex Mono**     | Elo, move lists, usernames, eyebrow labels. `.mp-mono`.               |

Scale:
- H1 pages: `text-[36–44px]` in `.mp-display` with `leading-[0.95–1]` and `tracking-tight`.
- H2 sections: `.mp-eyebrow` — 10px mono uppercase with `0.18em` tracking. Use instead of Tailwind's `text-sm uppercase tracking-wider`.
- Body copy: `text-[13–14px]` with `text-[var(--mp-ink-muted)]`.
- Display quotations / lede italic: `mp-display mp-italic`.

## Motion

Restraint is the rule. One choreographed page entrance, a few subtle hovers, no bouncy everything.

- **Page entry**: `.mp-enter` + `.mp-enter-1/2/3` staggered reveal (fade + 8px slide-up, cubic-bezier ease, 520ms). Apply to the two or three primary sections of each page.
- **Live pulse**: `.mp-livedot` — 6px dot breathing at 2s cadence. Used for live-match indicators, `agent_thinking` badge, ongoing post-match.
- **Hovers**: cards and rows lift via `.mp-panel-raised:hover` (surface + border transition). Links shift to `--mp-brass-bright` on hover.
- **Custom easing**: `--mp-ease: cubic-bezier(0.22, 0.61, 0.36, 1)` — snappy decel, not bouncy.

## Components

### Panel primitives
- `.mp-panel` — flat panel, 1px hairline border.
- `.mp-panel-raised` — hover-able card (lifts on hover).
- `.mp-framed` — adds brass corner marks (top-right + bottom-left by default; pair with `.mp-frame-tl` + `.mp-frame-br` spans for all four).

### Buttons
- `.mp-btn` + one of: `.mp-btn-brass` (primary, brushed-brass gradient with inset highlight), `.mp-btn-ghost` (outlined), `.mp-btn-danger` (oxblood, for resign/delete).

### Chips
- `.mp-chip` — 10px mono uppercase pill, used for rating badges, result chips, status.
- Rating chips use `--mp-rating-*` tokens.
- Result chips: `{{ result_chip(m) }}` macro renders Live/Resigned/Draw/White/Black with the right palette.

### Inputs
- `.mp-input` — engraved field. Brass border on focus. Pair with `.mp-eyebrow` labels.

### Board frame (play + watch)
- `.mp-board-frame` — padded brass-bordered wrapper around `chessboard.js`. Double-ruled via `::before` pseudo-element to evoke a framed set.

### Connection pill (play + watch)
- `.conn-pill` + `.conn-live` / `.conn-retry` / `.conn-lost` — uppercase 10px mono pills with semantic coloring.

### Chat bubbles (play + watch)
- `.chat-bubble-agent` — felt-green left border.
- `.chat-bubble-player` — brass left border, indented.
- `.chat-bubble-spectator` — blue-grey left border. `.in-transit` fades to 0.55 opacity (optimistic render).

### Jinja partials
Shared under `app/web/templates/_partials/`:
- `_macros.html` — `rating_badge`, `character_state_badge`, `visibility_badge`, `preset_badge`, `result_chip`.
- `character_card.html` — member-dossier card, used on `/`, `/discovery`, and player profile.
- `match_row.html` — live / recent match row on `/discovery`.

## Anti-patterns (deliberately avoided)

- **Inter / Roboto / system sans** — overused, generic. Fraunces + IBM Plex Sans is distinctive and recognisable.
- **Cool greys (`#737373` ilk)** — feel clinical. The palette uses warm ink (`#B5AD96`) that matches parchment.
- **Purple-blue gradient tiles on white** — textbook AI-slop. We use dark felt with brass accents instead.
- **Emoji-heavy icons** — restricted to `avatar_emoji` on characters (they're the character's portrait, framed) and the occasional `›` chevron. No sprinkled decorative emoji.
- **Animated everything** — motion is reserved for the page entry, live states, and hover. No confetti, no continuous ambient motion.

## Where the polish is applied

| Page                             | Level of polish                                                    |
| -------------------------------- | ------------------------------------------------------------------ |
| `/` (Characters)                 | Full — new header, card partial, brass rule, enter choreography.   |
| `/discovery`                     | Full — eyebrows, three sectioned surfaces, reused partials.        |
| `/matches/{id}` (Play)           | Full — board in brass frame, typography, thinking state, disconnect overlay. |
| `/matches/{id}/watch` (Spectate) | Full — read-only variant, crowd-noise panel with blue accent.      |
| `/matches/{id}/summary`          | Full — payoff screen: dossier-style result, italicised memories, brass accent on opponent note. |
| `/leaderboard/characters`        | Full — library-catalog tables, rank highlighted for top 3 in brass. |
| `/leaderboard/players`           | Full — same treatment, current user row highlighted.               |
| `/characters/{id}` (Detail)      | Header polished; hall-of-fame + memory sections on new tokens; body form sections inherit typography but still use Tailwind neutral classes. |
| `/characters/new`, `/edit`       | Light — new heading + brass submit button; form body inherits typography from base. |
| `/settings`                      | Full — panel groupings, brass radio accent.                        |
| `/login`, `/landing`             | Full — typographic hero, framed form, member-entrance eyebrow.     |
| `/players/{username}` (Profile)  | Full — dossier header, match rows, character card grid.            |

## Known gaps (for a follow-up polish)

- Character-detail memory ribbon section (body area below hero) still uses Tailwind neutral classes. Readable; out of palette.
- `new.html`/`edit.html` form body (inputs, fieldsets, sliders) not yet fully converted to `.mp-input` + `.mp-panel` idioms — the brass submit button + new header bring them into the system but the fields themselves retain the Phase 3a styling.
- `rating_hidden.html` (shown when a character exceeds the viewer's rating) untouched — low-traffic page.
- No responsive breakpoint review yet — the layout degrades gracefully on mobile thanks to Tailwind's `grid-cols-1` fallbacks but there's no dedicated mobile pass.
