import assert from "node:assert/strict"
import test from "node:test"
import type { Element, Root } from "hast"
import type { FullSlug } from "../../util/path"
import { defaultProcessedContent } from "../vfile"
import {
  generateReadingFeed,
  isReadingDigestSlug,
  readingFeedHtml,
  readingPublicationDate,
} from "./readingFeed"

const baseUrl = "shikicloud.github.io/MLSys"
const slug = "CN/reading/daily/2026-10-15"

test("only dated Chinese daily issues enter the feed", () => {
  assert.ok(isReadingDigestSlug(slug))
  assert.ok(isReadingDigestSlug("CN/reading/daily/2024-02-29"))
  for (const rejected of [
    "EN/reading/daily/2026-10-15",
    "CN/reading/index",
    "CN/reading/roadmap",
    "CN/reading/daily/index",
    "CN/reading/daily/2026-10-15/index",
    "CN/reading/daily/2026-02-29",
    "CN/reading/daily/2026-13-01",
    "CN/reading/daily/2026-10-15-extra",
  ]) {
    assert.equal(isReadingDigestSlug(rejected), false, rejected)
  }
})

test("publication dates stay fixed when an issue is edited or rebuilt", () => {
  assert.equal(
    readingPublicationDate(slug, {
      published: "2026-10-15T14:00:00+08:00",
      created: "2026-10-14",
      date: "2026-10-16",
      updated: "2026-12-01T23:00:00Z",
    }).toISOString(),
    "2026-10-15T06:00:00.000Z",
  )
  assert.equal(
    readingPublicationDate(slug, { updated: "2030-01-01" }).toISOString(),
    "2026-10-15T01:00:00.000Z",
  )
  assert.equal(
    readingPublicationDate(slug, { created: "2026-10-14" }).toISOString(),
    "2026-10-14T01:00:00.000Z",
  )
})

test("invalid metadata fails instead of silently changing the publication date", () => {
  for (const published of [
    "2026-02-30",
    "2026-02-30T09:00:00+08:00",
    "2026-10-15T09:00:00",
    "2026-10-15T25:00:00+08:00",
    "October 15, 2026",
    true,
  ]) {
    assert.throws(() => readingPublicationDate(slug, { published }), /publication/)
  }
})

test("full HTML resolves links and resources without changing the page tree", () => {
  const tree: Root = {
    type: "root",
    children: [
      ...[
        "../../../CN/reading/roadmap#kv-cache",
        "#contribution",
        "/CN/reading/",
        "/MLSys/CN/reading/",
        "https://arxiv.org/abs/1234.56789?x=1&y=2",
      ].map((href) => ({
        type: "element" as const,
        tagName: "a",
        properties: { href },
        children: [{ type: "text" as const, value: "Read" }],
      })),
      {
        type: "element",
        tagName: "img",
        properties: { src: "../../../assets/figure.png" },
        children: [],
      },
    ],
  }
  const original = structuredClone(tree)
  const html = readingFeedHtml(tree, slug, baseUrl)
  assert.ok(html.includes('href="https://shikicloud.github.io/MLSys/CN/reading/roadmap#kv-cache"'))
  assert.ok(
    html.includes(
      'href="https://shikicloud.github.io/MLSys/CN/reading/daily/2026-10-15#contribution"',
    ),
  )
  assert.equal(
    html.match(/href="https:\/\/shikicloud.github.io\/MLSys\/CN\/reading\/"/g)?.length,
    2,
  )
  assert.ok(html.includes('src="https://shikicloud.github.io/MLSys/assets/figure.png"'))
  assert.ok(html.includes("https://arxiv.org/abs/1234.56789?x=1&#x26;y=2"))
  assert.deepEqual(tree, original)
})

test("reader HTML omits website icons and duplicate title while preserving links and figures", () => {
  const svg = (properties: Element["properties"]): Element => ({
    type: "element", tagName: "svg", properties,
    children: [{ type: "element", tagName: "path", properties: { d: "M0 0L1 1" }, children: [] }],
  })
  const tree: Root = {
    type: "root",
    children: [
      { type: "text", value: "\n" },
      { type: "element", tagName: "h1", properties: {}, children: [{ type: "text", value: "Issue title" }] },
      {
        type: "element", tagName: "h2", properties: { id: "main" }, children: [
          { type: "text", value: "Main paper" },
          { type: "element", tagName: "a", properties: { role: "anchor", href: "#main" }, children: [svg({ width: 18 })] },
          { type: "element", tagName: "a", properties: { role: "anchor", href: "#main" }, children: [svg({ width: 18 })] },
        ],
      },
      {
        type: "element", tagName: "p", properties: {}, children: [
          { type: "element", tagName: "a", properties: { href: "https://example.org/paper" }, children: [
            { type: "text", value: "Original paper" },
            svg({ className: ["external-icon"], style: "max-width:0.8em" }),
            svg({ className: "external-icon" }),
            svg({ class: "external-icon extra" }),
          ] },
        ],
      },
      svg({ role: "img", ariaLabel: "Architecture figure", viewBox: "0 0 100 100" }),
      { type: "element", tagName: "img", properties: { src: "/assets/figure.png", alt: "Figure" }, children: [] },
    ],
  }
  const original = structuredClone(tree)
  const html = readingFeedHtml(tree, slug, baseUrl)
  assert.ok(!html.includes("<h1"))
  assert.ok(!html.includes("external-icon"))
  assert.ok(!html.includes('role="anchor"'))
  assert.ok(html.includes('<h2 id="main">Main paper</h2>'))
  assert.ok(html.includes('<a href="https://example.org/paper">Original paper</a>'))
  assert.equal(html.match(/<svg\b/g)?.length, 1, "content illustrations must survive")
  assert.ok(html.includes('aria-label="Architecture figure"'))
  assert.ok(html.includes('src="https://shikicloud.github.io/MLSys/assets/figure.png"'))
  assert.deepEqual(tree, original, "feed cleanup must not change the website")
})

test("RSS preserves complete HTML, escapes XML, limits issues, and excludes other pages", () => {
  const content = Array.from({ length: 31 }, (_, index) => {
    const day = (index + 1).toString().padStart(2, "0")
    const item = defaultProcessedContent({
      slug: `CN/reading/daily/2026-10-${day}` as FullSlug,
      frontmatter: {
        title: `Issue ${day} & <systems>`,
        description: "Summary & ]]> <text>",
        published: `2026-10-${day}T09:00:00+08:00`,
      },
    })
    item[0].children = [
      {
        type: "element",
        tagName: "p",
        properties: {},
        children: [{ type: "text", value: "Full body & ]]> 🦉\u0001" }],
      },
    ]
    return item
  })
  for (const excluded of [
    "CN/reading/index",
    "CN/reading/roadmap",
    "EN/reading/daily/2026-10-31",
  ]) {
    content.push(
      defaultProcessedContent({
        slug: excluded as FullSlug,
        frontmatter: { title: "EXCLUDED", published: "2026-12-01T09:00:00+08:00" },
      }),
    )
  }

  const xml = generateReadingFeed(baseUrl, content)
  assert.equal(xml.match(/<item>/g)?.length, 30)
  assert.ok(xml.includes("<title>Issue 31 &amp; &lt;systems&gt;</title>"))
  assert.ok(!xml.includes("<title>Issue 01"))
  assert.ok(!xml.includes("EXCLUDED"))
  assert.ok(xml.indexOf("<title>Issue 31") < xml.indexOf("<title>Issue 30"))
  assert.ok(
    xml.includes(
      '<guid isPermaLink="true">https://shikicloud.github.io/MLSys/CN/reading/daily/2026-10-31</guid>',
    ),
  )
  assert.ok(xml.includes("<pubDate>Sat, 31 Oct 2026 01:00:00 GMT</pubDate>"))
  assert.ok(xml.includes("<description>Summary &amp; ]]&gt; &lt;text&gt;</description>"))
  assert.ok(xml.includes("<content:encoded>&lt;p&gt;Full body"))
  assert.ok(xml.includes("🦉"))
  assert.ok(!xml.includes("<![CDATA["))
  assert.ok(!xml.includes("\u0001"))
  assert.ok(xml.includes('href="https://shikicloud.github.io/MLSys/reading.xml"'))
})
