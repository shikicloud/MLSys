# Daily reading: publishing contract

The daily reading archive is part of the existing Quartz site. Canonical content lives in `reading/digests/YYYY-MM-DD.json`; a deterministic generator renders the Chinese and English pages and their archive indexes. The dedicated RSS feed includes only Chinese daily editions, with stable issue links and publication dates. Existing wiki pages and the general site feed remain independent.

RSS exports semantic article HTML without the duplicate page title or Quartz heading/link icons. Readers may remove website styles, causing decorative SVG icons to render at large default sizes; article links and actual figures are preserved.

## Editorial policy

- Choose 3–5 directly relevant papers, technical reports, or substantive engineering analyses. Name one main read and make the others optional.
- There is no publication-age cutoff. Start from foundational work and follow prerequisite relationships; add newer work when it extends a concept already introduced.
- Prioritize well-established contributions, verified leading-venue publications, and strong research teams, while still judging evidence and relevance. Do not invent citation counts or equate affiliation with quality.
- Focus on agent workloads, rollout/training infrastructure, KV caching and scheduling, parallelism, and MoE systems. Exclude routine releases and promotional announcements.
- Verify title, paper year, venue, and source against the original paper or official proceedings. Distinguish preprint year from conference year.
- Explain the bottleneck, contribution, practical connection, prerequisites, reading advice, and a material limitation. Label measured results as author-reported and include their conditions when giving numbers.
- Write developed technical reading notes by default, following the structure and depth of the site's ThunderAgent and InferCept articles (`CN/wiki/llm-serving-for-agents/`). Do not wait for a follow-up question before explaining the mechanism. Their prose structure is a style reference, not a substitute for checking primary sources.
- The main reading needs a short overview, motivation and bottleneck, a walkthrough of the key mechanism, a concrete example or useful diagram, evaluation conditions and limitations, connections to the reader's public systems interests, and a suggested reading route. Define new terminology when first used. Optional readings should be shorter sectioned notes that still explain mechanisms and tradeoffs; a few summary bullets are insufficient.
- Distinguish original-paper results, current implementation facts, and our own analysis. Label illustrative numbers as examples. Pin code links to a verified commit and state the verification date. Do not describe current documentation as evidence that a configuration has been run or benchmarked.
- An existing wiki page or previous recommendation is not evidence that the reader has read or mastered a paper. Avoid repeating recommended papers unless the new entry includes an explicit bilingual `revisitReason`.
- Keep personal resumes, contacts, internal work details, cluster paths, and notification credentials out of this public repository.

## Add an edition

1. Read this file, `reading/preferences.json`, the roadmap, and existing `reading/digests/` records. Respect explicitly completed reading and use existing IDs for previously recommended papers. For source-code explanations, pin links to a verified commit and distinguish model-specific implementations.
2. Create one JSON record following `2026-10-15.json`, with bilingual sectioned prose as described below. Use the actual Beijing date and first publication timestamp (`+08:00`). Do not predate missed editions or change `publishedAt` when correcting one.
3. Run from the repository root:

   ```sh
   node scripts/reading-digest.mjs --write
   node --test scripts/reading-digest.test.mjs
   node scripts/reading-digest.mjs --check
   ```

4. Review the generated CN and EN editions and archive indexes. Append a concise note to each language's `log.md`; do not rewrite historical entries. Change only digest data, generated reading pages, and those append-only log entries during routine runs.
5. Commit the complete edition as one change to `main` using the authorized GitHub connection. Check the current branch head first; use a fast-forward / expected-head guard and refresh on conflicts. Never overwrite an unrelated concurrent change. A failed publication must remain visible as a failure, not be reported as synchronized.
6. The existing Pages workflow validates the data and feed, builds Quartz, and deploys the website. Check its result and the live edition before reporting publication success. A successful commit alone is not a successful deployment.

`--check` validates all source records and verifies that committed generated pages match them. Do not hand-edit generated daily pages or reading indexes. Roadmap pages are maintained separately.

## Sectioned article format

Keep the existing five summary fields (`contribution`, `relevance`, `reading`, `prerequisites`, `caveat`) in each paper's `zh` and `en` objects. New editions also include `sections` in both languages, with matching section counts and equivalent coverage:

```json
"sections": [
  {
    "heading": "Why this bottleneck matters",
    "paragraphs": [
      "A complete explanatory paragraph with **inline emphasis** and a [primary source](https://example.org/paper).",
      "A concrete example or a clearly labeled inference connecting the mechanism to a systems decision."
    ]
  }
]
```

Each heading and paragraph is one nonempty line. Inline Markdown links, emphasis, code spans, and image references are supported; raw HTML, control characters, and block Markdown inside these strings are not. Sectioned output shows paper metadata and reading advice, then real subsection headings and paragraphs. Integrate contribution, relevance, prerequisites, and limitations into the prose rather than repeating the summary fields as a checklist. Historical records without sections retain their original rendering.

The full sectioned Chinese article is included in RSS. Use ordinary, bounded raster images for diagrams that must work in readers; do not rely on JavaScript, site CSS, or inline SVG for understanding the article. Make an illustration only when it clarifies a mechanism. If an edition needs a new figure asset, include that referenced asset and its editable source in the same commit as the edition.

## Mobile reading

- Archive: <https://shikicloud.github.io/MLSys/CN/reading/>
- Dedicated feed: <https://shikicloud.github.io/MLSys/reading.xml>
- English archive: <https://shikicloud.github.io/MLSys/EN/reading/>

Subscribe to the dedicated feed in a mobile RSS reader. Content is shared across devices; read/unread status and notifications are managed by the chosen reader. The website does not claim to synchronize personal reading completion.

The existing Codex task starts at 09:00 Asia/Shanghai. Research and deployment take time, so this is a start time rather than a delivery deadline. The local generation task needs the computer and app available; the published site is independently hosted by GitHub. This repository does not contain an unattended cloud model runner or a model API key.

Direct push notifications are optional and require a chosen channel. RSS polling is reader-dependent. GitHub Mobile does not treat ordinary commits or Pages deploys as paper notifications. Bark requires an iPhone, notification permission, and a private device key; never put that key in source files, RSS, URLs shown publicly, or build logs.

## Deployment detail

Normal commits from the authorized user / GitHub App connection trigger the Pages push workflow. If generation is later moved into GitHub Actions, a commit made with that workflow's `GITHUB_TOKEN` does not automatically trigger a second push workflow. Build and deploy in that same run, or explicitly dispatch the deployment workflow. Do not add a second daily generator while the existing scheduled task is active.

References: [GitHub workflow triggers](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow), [GitHub notification settings](https://docs.github.com/en/subscriptions-and-notifications/get-started/configuring-notifications), [RSS specification](https://www.rssboard.org/rss-2-0-11).
