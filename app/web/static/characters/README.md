# Per-character static assets

Each preset grandmaster has a folder here; drop assets in by filename and
the room pages + play page will pick them up with no code changes.

```
characters/
  viktor_petrov/
    ambient.mp3            # ambient bed, looped
    background.webp        # still used as the room backdrop
    emotions/
      neutral.mp4
      pleased.mp4
      annoyed.mp4
      excited.mp4
      focused.mp4
      uncertain.mp4
      smug.mp4
      deflated.mp4
  margot_lindqvist/
    ambient.mp3
    background.webp
    emotions/              # optional — if empty, text indicator takes over
  kenji_sato/
    ambient.mp3
    background.webp
    emotions/
  archibald_finch/
    ambient.mp3
    background.webp
    emotions/
```

## Notes

- **Audio autoplay** is blocked by browsers until the user interacts.
  The room page starts the ambient track muted and unmutes on click of
  the header audio toggle.
- **Emotion clips** are non-seamless loops by design (they're cut from
  user-recorded footage with varying start frames). The player
  cross-fades between clips over ~420 ms so the joins stay soft.
- **Missing files** degrade gracefully: if `neutral.mp4` 404s, the
  whole video stage hides and the text-only emotion indicator
  (`neutral ●●○` etc.) takes over.
- **Custom player-authored characters** get the default (no theme,
  no ambient, no video) — themes are preset-only until we add a
  character-level override column.

## Mapping to the Soul emotion vocabulary

The Soul LLM emits exactly these eight emotions
(`app/schemas/agents.py::Emotion`):

    neutral, pleased, annoyed, excited, focused, uncertain, smug, deflated

If your source clips use different labels (e.g. smiling / laughing /
bored / angry), either rename the files or adjust `emotion_clips` in
`app/characters/rooms.py::VIKTOR_ROOM`.
