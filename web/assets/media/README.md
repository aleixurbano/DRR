# Recovery media (media panel, top of the page)

The media panel of `index.html` expects this file here. Drop it in and the page
picks it up with no code change:

- `recovery.mp4` — a recorded episode recovery (or several joined into one clip).
- `recovery_poster.jpg` — a still frame shown before the video plays (optional
  but recommended for a clean first paint).

The recovery trace (task, failure reason, step-by-step log) is real text in the
`.trace-panel` markup in `index.html`, not an image, so it can be edited
directly there.

If you use a different filename or extension for the video, update the
`<source src>` in the `.media-grid` block of `index.html`.
