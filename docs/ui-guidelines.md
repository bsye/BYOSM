# UI guidelines

Rules for every page of the web app in `ui/`. The look is standard shadcn: the default `radix-nova` style, the
neutral base colour and Geist, exactly as `npx shadcn init` and the stock components ship them. We change how pages
are composed, never how components look.

Sources: Anthropic's `frontend-design` skill (plugin `frontend-design@claude-code-plugins`) for what reads as
AI-generated, the `ux-skill` dashboard rules, and shadcn's own `dashboard-01` block for layout.

## Components

- Only stock shadcn components from `ui/src/components/ui`, added with the shadcn CLI and never edited. Missing one?
  `npx shadcn@4.20.0 add <name>`.
- No custom colours, radii, shadows or fonts. Colours come from theme tokens (`bg-muted`, `text-muted-foreground`,
  `var(--chart-1)`), never hex values or arbitrary Tailwind colours.
- Charts use the shadcn `chart` component the way its examples do: `accessibilityLayer`, `cursor={false}` tooltips,
  no y axis unless the scale matters, `tickMargin` on the x axis, `radius` on bars.

## Layout

- Every page uses the same frame: the inset sidebar, the header with breadcrumbs, then a column with `gap-4 md:gap-6`
  inside `@container/main`. Never add page-level padding of your own.
- Inverted pyramid: status first (is it good?), then trends and comparisons, then details (tables, logs, rows).
- Cards in the same grid row have the same height (`h-full` on the card, grid `items-stretch`), the same internal
  structure (header, content, footer in the same order) and the same padding.
- One primary number per page. It is visibly larger and heavier than everything else (`text-3xl font-semibold`);
  secondary numbers are `text-2xl`; labels are `text-sm text-muted-foreground`. The hierarchy must survive grayscale.
- Two to four secondary metrics at most. A fifth is noise: move it to a table.
- Cards group related things. No card holding a single word, no card inside a card, no wrapper without a purpose.
- One density per page. Tables use the stock row height; do not mix compact and comfortable lists.

## Text

- Every label says what the number is and against what: "Test accuracy on Banking77", not "Score". Units and
  splits are always named (test, calib, ms per request).
- Sentence case. No ALL-CAPS eyebrows, no middle-dot meta strings ("A · B · C"), no arrows appended to buttons.
- Buttons name the outcome: "Build it", "Save changes", "Run this proposal". The confirmation uses the same verb.
- Descriptions are one sentence, plain, and say what the thing does for the user. No selling.
- Errors say what happened and what to do next. Empty states invite the next action.

## Numbers

- `tabular-nums` on every number. Monospace only for code: commands, logs, IDs, spec files, tool calls.
- Scores to three decimals, with the interval when one exists: `0.937 [0.928, 0.948]`.
- Colour never carries meaning alone: a delta has a sign and an icon as well as a colour.

## States

- Every region that loads data has a skeleton of the same size, an empty state with the next action, and an error
  state that names the failure.
- Live indicators only for things that really change in real time (running jobs, the WebSocket), at most two per
  screen.

## Motion and decoration

- No entrance animations on sections and no hover effects on every card. Motion only answers a user action or shows
  live data changing.
- Badges encode a state (promoted, rival, exported, running). They are not decoration; at most three per row.

## Before a page ships

1. Take a screenshot in light and dark mode at desktop and phone width.
2. Check each rule above; fix what fails.
3. Remove one element that does not earn its place.
