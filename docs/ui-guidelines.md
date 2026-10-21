# UI guidelines

Rules for every page of the web app in `ui/`. The look is standard shadcn: the default `radix-nova` style and the
neutral base colour, exactly as `npx shadcn init` and the stock components ship them. The one exception is the
typeface: Hanken Grotesk, set only through the `--font-sans` token in `ui/src/index.css`. We change how pages are
composed, never how components look.

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
  secondary numbers are `text-2xl`; labels follow the text levels below. The hierarchy must survive grayscale.
- Two to four secondary metrics at most. A fifth is noise: move it to a table.
- Cards group related things. No card holding a single word, no card inside a card, no wrapper without a purpose.
- One density per page. Tables use the stock row height; do not mix compact and comfortable lists.

## Text levels

Every card and section uses exactly these levels, top to bottom. Two neighbouring texts never share a level unless
they are the same kind of thing.

| level | what | classes |
|---|---|---|
| title | the thing's name | `CardTitle` (stock) |
| context | one line of description, clamped | `CardDescription line-clamp-1` |
| hero | the one number or state the card exists for | `text-3xl font-semibold tabular-nums` |
| label | only when the title does not already say what the hero is; otherwise put it in the hero's `title` tooltip | `text-xs text-muted-foreground` |
| detail | the comparison or note under the hero: "+0.129 vs Jev" | `text-xs`, the figure `text-foreground font-medium`, the rest muted |

- Hierarchy comes from opacity as much as from size. Four steps, and only these: `text-foreground` for what the
  reader came for, `text-foreground/70` for supporting values (params, latency, counts), `text-muted-foreground` for
  labels, `text-muted-foreground/70` for the least important (intervals, units, ids). In a table the key column and
  the main score are full strength; every other column steps down.
- Fewer levels beat more: drop any line the reader does not need to decide something. A label that repeats what the
  page already says is noise.
- No decorative icons next to the product name or headings; icons belong on buttons and on state.
- The hero is the only large text in a card. Labels and details are small and quiet so it stands out without colour.
- A description never wraps to two lines inside a card; the full text belongs on the detail page.
- Human names, never internal codes: the benchmark's dataset name (`banking77`), not the suite id (`b77`); the rival's
  short name (`Jev`, `bert-base-NER`), not `hf:org/repo`.

## Alignment

- Cards in a grid row share one height, and their parts line up across the row: labels clamp to one line so every
  header ends at the same height, and the footer is pinned to the bottom (`mt-auto`) so no empty band appears under it.
- Numbers in the same position on neighbouring cards sit on the same baseline.

## Less text

- First glance wins: a user who has never seen BYOSM must know what each page is for, and what to click, in five
  seconds. Use the archetype the user already knows (list, detail, tabs, stat cards, table, prompt bar).
- Default to no `CardDescription`. Add one only when the title alone is ambiguous, and keep it under ten words.
- Labels are one to three words: "Test accuracy", "vs Jev", "Typed before firing". No sentences in labels.
- Methodology (calib vs test, how a rival is chosen, what a gate is) lives in an info tooltip or the docs, not on screen.
- Count the words on a page before shipping; if a section has more words than numbers, cut.

## Text

- Every label says what the number is and against what: "Test accuracy on Banking77", not "Score". Units and
  splits are always named (test, calib, ms per request).
- Sentence case. No ALL-CAPS eyebrows, no middle-dot meta strings ("A · B · C"), no arrows appended to buttons.
- Every button has a lucide icon before its label (`<Hammer /> Build`). Icon-only buttons have an `aria-label` and a
  tooltip. One primary button per view (`variant="default"`), the rest `outline`; destructive actions use the
  `destructive` variant and confirm first.
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
