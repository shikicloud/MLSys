import type { Root } from "hast"
import { toHtml } from "hast-util-to-html"
import { SKIP, visit } from "unist-util-visit"
import type { FullSlug } from "../../util/path"
import type { QuartzEmitterPlugin } from "../types"
import type { ProcessedContent } from "../vfile"
import { write } from "./helpers"

const feedTitle = "MLSys 每日论文精选"
const feedDescription = "Agent serving 与 RL infra：经典论文、系统设计与值得精读的新工作。"
const digestPrefix = "CN/reading/daily/"

function isCalendarDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false
  const date = new Date(`${value}T00:00:00Z`)
  return Number.isFinite(date.getTime()) && date.toISOString().slice(0, 10) === value
}

export function isReadingDigestSlug(slug: string): boolean {
  return slug.startsWith(digestPrefix) && isCalendarDate(slug.slice(digestPrefix.length))
}

/** Never use filesystem dates, updated, or build time: edits must not republish an issue. */
export function readingPublicationDate(
  slug: string,
  frontmatter: Record<string, unknown> = {},
): Date {
  if (!isReadingDigestSlug(slug)) throw new Error(`Invalid reading digest slug: ${slug}`)

  const value = frontmatter.published ?? frontmatter.created ?? frontmatter.date
  if (value === undefined) return new Date(`${slug.slice(digestPrefix.length)}T09:00:00+08:00`)
  if (typeof value !== "string") {
    throw new Error(`Reading digest ${slug} needs a string publication date`)
  }

  // Date-only metadata follows the digest's Asia/Shanghai morning schedule.
  if (isCalendarDate(value)) return new Date(`${value}T09:00:00+08:00`)
  const timestamp =
    /^\d{4}-\d{2}-\d{2}T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d{1,3})?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)$/
  if (!timestamp.test(value) || !isCalendarDate(value.slice(0, 10))) {
    throw new Error(`Reading digest ${slug} needs an ISO publication timestamp with a timezone`)
  }
  return new Date(value)
}

function escapeXml(value: string): string {
  return value
    .replace(/[^\u0009\u000A\u000D\u0020-\uD7FF\uE000-\uFFFD\u{10000}-\u{10FFFF}]/gu, "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;")
}

function siteRoot(baseUrl: string): URL {
  return new URL(`https://${baseUrl.replace(/\/+$/, "")}/`)
}

function absoluteUrl(value: string, pageUrl: URL, root: URL): string {
  if (/^[a-z][a-z\d+.-]*:/i.test(value)) return value
  if (value.startsWith("/") && !value.startsWith("//")) {
    // Quartz normally emits relative URLs; also support site-root links in embedded HTML.
    if (value === root.pathname.slice(0, -1) || value.startsWith(root.pathname)) {
      return new URL(value, root).href
    }
    return new URL(value.slice(1), root).href
  }
  return new URL(value, pageUrl).href
}

export function readingFeedHtml(tree: Root, slug: string, baseUrl: string): string {
  const root = siteRoot(baseUrl)
  const pageUrl = new URL(slug, root)
  const copy = structuredClone(tree)
  // Feed readers display the issue title separately. Website CSS hides this duplicate H1.
  const firstElement = copy.children.findIndex((node) => node.type === "element")
  const first = copy.children[firstElement]
  if (first?.type === "element" && first.tagName === "h1") copy.children.splice(firstElement, 1)

  visit(copy, "element", (node, index, parent) => {
    const classValue = node.properties.className ?? node.properties.class
    const classes = Array.isArray(classValue)
      ? classValue
      : typeof classValue === "string"
        ? classValue.split(/\s+/)
        : []
    const decorative =
      (node.tagName === "a" && node.properties.role === "anchor") ||
      (node.tagName === "svg" && classes.includes("external-icon"))
    if (decorative && parent && index !== undefined) {
      parent.children.splice(index, 1)
      return [SKIP, index]
    }
    for (const attribute of ["href", "src", "poster", "cite"]) {
      const value = node.properties[attribute]
      if (typeof value === "string") {
        node.properties[attribute] = absoluteUrl(value, pageUrl, root)
      }
    }
  })
  return toHtml(copy)
}

export function generateReadingFeed(baseUrl: string, content: ProcessedContent[]): string {
  const root = siteRoot(baseUrl)
  const entries = content
    .filter(([, file]) => isReadingDigestSlug(file.data.slug ?? ""))
    .map(([tree, file]) => ({
      tree,
      file,
      slug: file.data.slug!,
      published: readingPublicationDate(file.data.slug!, file.data.frontmatter),
    }))
    .sort((a, b) => b.published.getTime() - a.published.getTime() || b.slug.localeCompare(a.slug))
    .slice(0, 30)

  const items = entries.map(({ tree, file, slug, published }) => {
    const url = new URL(slug, root).href
    const title = file.data.frontmatter?.title ?? slug.slice(digestPrefix.length)
    const description = file.data.frontmatter?.description ?? file.data.description ?? ""
    return `<item>
      <title>${escapeXml(title)}</title>
      <link>${escapeXml(url)}</link>
      <guid isPermaLink="true">${escapeXml(url)}</guid>
      <pubDate>${published.toUTCString()}</pubDate>
      <description>${escapeXml(description)}</description>
      <content:encoded>${escapeXml(readingFeedHtml(tree, slug, baseUrl))}</content:encoded>
    </item>`
  })

  return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>${escapeXml(feedTitle)}</title>
    <link>${escapeXml(new URL("CN/reading/", root).href)}</link>
    <description>${escapeXml(feedDescription)}</description>
    <language>zh-CN</language>
    <atom:link href="${escapeXml(new URL("reading.xml", root).href)}" rel="self" type="application/rss+xml"/>
    ${items.join("\n    ")}
  </channel>
</rss>`
}

export const ReadingFeed: QuartzEmitterPlugin = () => ({
  name: "ReadingFeed",
  async *emit(ctx, content) {
    const baseUrl = ctx.cfg.configuration.baseUrl
    if (!baseUrl) throw new Error("ReadingFeed requires configuration.baseUrl")
    yield write({
      ctx,
      content: generateReadingFeed(baseUrl, content),
      slug: "reading" as FullSlug,
      ext: ".xml",
    })
  },
  externalResources: (ctx) => ({
    additionalHead: [
      <link
        rel="alternate"
        type="application/rss+xml"
        title={feedTitle}
        href={new URL("reading.xml", siteRoot(ctx.cfg.configuration.baseUrl!)).href}
      />,
    ],
  }),
})
