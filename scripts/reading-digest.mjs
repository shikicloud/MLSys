#!/usr/bin/env node
import { lstat, mkdir, readFile, readdir, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const languages = ['zh', 'en'];
const fields = ['contribution', 'relevance', 'reading', 'prerequisites', 'caveat'];
const defaultRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const feedUrl = 'https://shikicloud.github.io/MLSys/reading.xml';
const copy = {
  zh: {
    folder: 'CN', title: '每日论文推荐', description: '从经典基础到新研究：agent serving 与 RL infra 的每日阅读候选。',
    introduction: '每天推荐 1 篇主读与 2–4 篇选读；候选不等于已读，无需一天读完。优先建立系统基础，再逐步连接当前的 agent serving 与 RL infra 工作。',
    sync: '本页与每日归档随仓库发布同步；手机可浏览网站或订阅 RSS。RSS 订阅是否推送通知取决于所用阅读器。',
    roadmap: '学习路线', feed: '订阅 RSS', archive: '历史推荐（最新在前）', main: '主读', optional: '选读', source: '原文',
    labels: ['核心贡献', '与当前工作的关系', '阅读建议', '前置知识', '局限与注意'], venue: '年份 / 发表信息', revisit: '再次推荐的原因', back: '全部推荐',
  },
  en: {
    folder: 'EN', title: 'Daily reading recommendations', description: 'Daily candidates for agent serving and RL infrastructure, from foundational classics to new research.',
    introduction: 'Each digest recommends 1 main paper and 2–4 optional papers. A recommendation does not mean it has been read, and there is no need to finish everything in one day. Build the foundations, then connect them to current agent serving and RL infrastructure work.',
    sync: 'This index and the daily archive sync with repository publishing. Browse on your phone or subscribe through RSS; notification delivery depends on your RSS reader.',
    roadmap: 'Learning roadmap', feed: 'Subscribe via RSS', archive: 'Archive (newest first)', main: 'Main reading', optional: 'Optional reading', source: 'Original source',
    labels: ['Core contribution', 'Connection to current work', 'Reading guidance', 'Prerequisites', 'Limitations'], venue: 'Year / publication', revisit: 'Why revisit', back: 'All recommendations',
  },
};

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function isDate(value) {
  return typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value)
    && Number.isFinite(Date.parse(`${value}T00:00:00Z`))
    && new Date(`${value}T00:00:00Z`).toISOString().slice(0, 10) === value;
}

function text(value, label) {
  assert(typeof value === 'string' && value.trim().length > 0, `${label} must be nonempty text`);
  assert(!/[\u0000-\u001f\u007f]/.test(value), `${label} must be a single Markdown paragraph without control characters`);
  assert(!/<(?:[a-zA-Z!/?])/.test(value), `${label} must not contain HTML`);
}

function bilingual(value, names, label) {
  assert(value && typeof value === 'object', `${label} must be bilingual`);
  for (const lang of languages) {
    if (names) {
      assert(value[lang] && typeof value[lang] === 'object', `${label}.${lang} is required`);
      for (const name of names) text(value[lang][name], `${label}.${lang}.${name}`);
    } else text(value[lang], `${label}.${lang}`);
  }
}

function canonicalUrl(value) {
  const url = new URL(value);
  url.hash = '';
  return url.href;
}

function inlineParagraph(value, label) {
  text(value, label);
  // Each string becomes a standalone paragraph. In particular, an open fence
  // would consume later sections and papers even though the string has no newline.
  const block = /^ {4}/.test(value)
    || /^ {0,3}(?:#{1,6}(?: |$)|>|[-+*](?: |$)|\d{1,9}[.)](?: |$))/.test(value)
    || /^ {0,3}(?:`{3,}[^`]*|~{3,}.*|\${2,}[^$]*)$/.test(value)
    || /^ {0,3}(?:(?:\* *){3,}|(?:- *){3,}|(?:_ *){3,})$/.test(value)
    || /^ {0,3}\[(?:\\.|[^\]\\])+\]:/.test(value);
  assert(!block, `${label} must contain inline Markdown only, not block syntax`);
}

function validateSections(paper, label) {
  if (languages.every(lang => paper[lang].sections === undefined)) return;
  const { sections: zh } = paper.zh;
  const { sections: en } = paper.en;
  assert(Array.isArray(zh) && Array.isArray(en), `${label}.sections must be arrays in both zh and en`);
  assert(zh.length > 0 && en.length > 0, `${label}.sections must be nonempty`);
  assert(zh.length === en.length, `${label}.sections must have matching zh and en section counts`);
  for (const lang of languages) {
    paper[lang].sections.forEach((section, index) => {
      const location = `${label}.${lang}.sections[${index}]`;
      assert(section && typeof section === 'object' && !Array.isArray(section), `${location} must be an object`);
      text(section.heading, `${location}.heading`);
      assert(Array.isArray(section.paragraphs) && section.paragraphs.length > 0, `${location}.paragraphs must be a nonempty array`);
      section.paragraphs.forEach((paragraph, paragraphIndex) => inlineParagraph(paragraph, `${location}.paragraphs[${paragraphIndex}]`));
    });
  }
}

function validateDigest(digest, filename, now) {
  assert(digest && digest.schemaVersion === 1, `${filename}: schemaVersion must be 1`);
  assert(isDate(digest.date) && filename === `${digest.date}.json`, `${filename}: invalid date or date does not match filename`);
  const stamp = digest.publishedAt;
  assert(typeof stamp === 'string' && /^\d{4}-\d{2}-\d{2}T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d{1,3})?\+08:00$/.test(stamp), `${filename}: publishedAt must be ISO time with +08:00`);
  assert(isDate(stamp.slice(0, 10)) && Number.isFinite(Date.parse(stamp)), `${filename}: invalid publishedAt`);
  const shanghaiDate = new Date(Date.parse(stamp) + 8 * 3600_000).toISOString().slice(0, 10);
  assert(shanghaiDate === digest.date, `${filename}: publishedAt must fall on the digest date in Asia/Shanghai`);
  assert(Date.parse(stamp) <= now.getTime(), `${filename}: future publication is not allowed`);
  bilingual(digest, ['title', 'summary'], filename);
  assert(Array.isArray(digest.papers) && digest.papers.length >= 3 && digest.papers.length <= 5, `${filename}: expected 3–5 papers`);
  assert(digest.papers.filter(p => p?.role === 'main').length === 1, `${filename}: expected exactly one main paper`);
  const ids = new Set();
  const urls = new Set();
  for (const paper of digest.papers) {
    assert(paper && typeof paper === 'object', `${filename}: invalid paper`);
    const label = `${filename}: paper ${paper.id ?? '?'}`;
    assert(typeof paper.id === 'string' && paper.id.length <= 80 && /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(paper.id), `${label}: unsafe paper ID`);
    assert(!ids.has(paper.id), `${label}: duplicate ID within a digest`);
    ids.add(paper.id);
    assert(paper.role === 'main' || paper.role === 'optional', `${label}: invalid role`);
    assert(Number.isInteger(paper.year) && paper.year >= 1940 && paper.year <= Number(digest.date.slice(0, 4)), `${label}: invalid publication year`);
    text(paper.title, `${label}.title`);
    text(paper.venue, `${label}.venue`);
    assert(typeof paper.url === 'string' && /^https:\/\//.test(paper.url) && !/[\s<>\\]/.test(paper.url), `${label}: URL must be an absolute HTTPS URL`);
    let url;
    try { url = new URL(paper.url); } catch { throw new Error(`${label}: invalid URL`); }
    assert(url.protocol === 'https:' && url.hostname && !url.username && !url.password, `${label}: URL must be HTTPS without credentials`);
    const normalizedUrl = canonicalUrl(paper.url);
    assert(!urls.has(normalizedUrl), `${label}: duplicate source URL within a digest: ${normalizedUrl}`);
    urls.add(normalizedUrl);
    bilingual(paper, fields, label);
    validateSections(paper, label);
    if (paper.revisitReason !== undefined) bilingual(paper.revisitReason, null, `${label}.revisitReason`);
  }
}

// Refuse symlinks at every managed path component, including existing outputs.
async function safePath(root, relative, allowMissing = false) {
  const parts = relative.split('/');
  assert(!path.isAbsolute(relative) && parts.every(p => p && p !== '.' && p !== '..'), `Unsafe path: ${relative}`);
  let current = root;
  for (let index = 0; index < parts.length; index++) {
    current = path.join(current, parts[index]);
    let info;
    try { info = await lstat(current); } catch (error) {
      if (allowMissing && error.code === 'ENOENT') return current + (index + 1 < parts.length ? path.sep + parts.slice(index + 1).join(path.sep) : '');
      throw error;
    }
    assert(!info.isSymbolicLink(), `Unsafe symlink: ${current}`);
    assert(index === parts.length - 1 || info.isDirectory(), `Expected directory: ${current}`);
  }
  return current;
}

async function checkOrphanPages(root, dates) {
  const orphans = [];
  for (const { folder } of Object.values(copy)) {
    const directory = await safePath(root, `${folder}/reading/daily`, true);
    let entries;
    try { entries = await readdir(directory); } catch (error) {
      if (error.code === 'ENOENT') continue;
      throw error;
    }
    for (const name of entries.sort()) {
      if (/^\d{4}-\d{2}-\d{2}\.md$/.test(name) && !dates.has(name.slice(0, -3))) {
        orphans.push(path.join(directory, name));
      }
    }
  }
  assert(!orphans.length, `Daily pages have no matching digest JSON; restore the sources or review and remove the pages manually:\n${orphans.join('\n')}`);
}

const yaml = value => JSON.stringify(value);
const label = value => value.replace(/[\\[\]|]/g, '\\$&');

function metadata({ title, description, date, publishedAt, created = date, ids }) {
  return [
    '---', `title: ${yaml(title)}`, 'category: ml-sys', 'tags: [reading-digest]',
    `description: ${yaml(description)}`, `published: ${yaml(publishedAt)}`, `date: ${date}`,
    `created: ${created}`, `updated: ${date}`, 'status: growing',
    `paper_ids: ${JSON.stringify(ids)}`, '---', '',
  ].join('\n');
}

function renderDigest(digest, lang) {
  const c = copy[lang];
  const lines = [metadata({ ...digest[lang], description: digest[lang].summary, date: digest.date, publishedAt: digest.publishedAt, ids: digest.papers.map(p => p.id) }),
    `# ${digest[lang].title}`, '', digest[lang].summary, '', c.introduction, '',
    `[[${c.folder}/reading/index|${c.back}]] · [[${c.folder}/reading/roadmap|${c.roadmap}]] · [${c.feed}](${feedUrl})`, '',
  ];
  const ordered = [...digest.papers].sort((a, b) => Number(b.role === 'main') - Number(a.role === 'main'));
  for (const paper of ordered) {
    lines.push(`## ${paper.role === 'main' ? c.main : c.optional} · ${paper.title}`, '');
    const info = `**${c.venue}：** ${paper.year} · ${paper.venue} · [${c.source}](<${paper.url}>)`;
    const sections = paper[lang].sections;
    if (sections) {
      lines.push(info, '', `**${c.labels[2]}：** ${paper[lang].reading}`, '');
      if (paper.revisitReason) lines.push(`**${c.revisit}：** ${paper.revisitReason[lang]}`, '');
      for (const section of sections) {
        lines.push(`### ${section.heading}`, '');
        for (const paragraph of section.paragraphs) lines.push(paragraph, '');
      }
    } else {
      lines.push(`- ${info}`);
      fields.forEach((field, i) => lines.push(`- **${c.labels[i]}：** ${paper[lang][field]}`));
      if (paper.revisitReason) lines.push(`- **${c.revisit}：** ${paper.revisitReason[lang]}`);
      lines.push('');
    }
  }
  return lines.join('\n');
}

function renderIndex(digests, lang) {
  const c = copy[lang];
  const latest = digests[0];
  const lines = [metadata({ title: c.title, description: c.description, date: latest.date, publishedAt: latest.publishedAt, created: digests.at(-1).date, ids: [...new Set(digests.flatMap(d => d.papers.map(p => p.id)))] }),
    `# ${c.title}`, '', c.introduction, '', c.sync, '',
    `[[${c.folder}/reading/roadmap|${c.roadmap}]] · [${c.feed}](${feedUrl})`, '', `## ${c.archive}`, '',
  ];
  for (const digest of digests) {
    const main = digest.papers.find(p => p.role === 'main');
    lines.push(`- [[${c.folder}/reading/daily/${digest.date}|${digest.date} · ${label(digest[lang].title)}]] — ${c.main}: ${main.title}`);
  }
  return lines.join('\n') + '\n';
}

export async function generate({ root = defaultRoot, mode = 'check', now = new Date() } = {}) {
  assert(mode === 'check' || mode === 'write', 'Mode must be check or write');
  assert(now instanceof Date && Number.isFinite(now.getTime()), 'Invalid current time');
  root = await realpath(root);
  assert((await lstat(root)).isDirectory(), 'Root must be a directory');
  const input = await safePath(root, 'reading/digests');
  const entries = await readdir(input, { withFileTypes: true });
  const filenames = entries.filter(e => e.name.endsWith('.json')).map(e => e.name).sort();
  const digests = [];
  const seen = new Set();
  const seenUrls = new Set();
  for (const filename of filenames) {
    assert(/^\d{4}-\d{2}-\d{2}\.json$/.test(filename), `Invalid digest filename: ${filename}`);
    const source = await safePath(root, `reading/digests/${filename}`);
    let digest;
    try { digest = JSON.parse(await readFile(source, 'utf8')); } catch (error) { throw new Error(`${filename}: ${error.message}`); }
    validateDigest(digest, filename, now);
    for (const paper of digest.papers) {
      assert(!seen.has(paper.id) || paper.revisitReason, `${filename}: ${paper.id} was already recommended; bilingual revisitReason is required`);
      const normalizedUrl = canonicalUrl(paper.url);
      assert(!seenUrls.has(normalizedUrl) || paper.revisitReason, `${filename}: source URL was already recommended under this or another ID; bilingual revisitReason is required: ${normalizedUrl}`);
      seen.add(paper.id);
      seenUrls.add(normalizedUrl);
    }
    digests.push(digest);
  }
  await checkOrphanPages(root, new Set(digests.map(d => d.date)));
  assert(digests.length > 0, 'No digest JSON files found in reading/digests');
  digests.reverse();
  const outputs = new Map();
  for (const lang of languages) {
    const folder = copy[lang].folder;
    for (const digest of digests) outputs.set(`${folder}/reading/daily/${digest.date}.md`, renderDigest(digest, lang));
    outputs.set(`${folder}/reading/index.md`, renderIndex(digests, lang));
  }
  // Validate all source records and destination paths before writing anything.
  for (const destination of outputs.keys()) await safePath(root, destination, true);
  const mismatches = [];
  for (const [relative, content] of outputs) {
    const destination = path.join(root, relative);
    let existing;
    try { existing = await readFile(destination, 'utf8'); } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    if (existing === content) continue;
    if (mode === 'check') mismatches.push(relative);
    else {
      await mkdir(path.dirname(destination), { recursive: true });
      await writeFile(destination, content, 'utf8');
    }
  }
  assert(!mismatches.length, `Generated files are missing or outdated; run node scripts/reading-digest.mjs --write:\n${mismatches.join('\n')}`);
  return { digests: digests.length, files: outputs.size };
}

async function main(args) {
  let mode;
  let root = defaultRoot;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--write' || args[i] === '--check') {
      assert(!mode, 'Choose exactly one of --write or --check');
      mode = args[i].slice(2);
    } else if (args[i] === '--root') {
      assert(args[i + 1] && !args[i + 1].startsWith('--'), '--root requires a directory');
      root = path.resolve(args[++i]);
    } else throw new Error(`Unknown argument: ${args[i]}`);
  }
  assert(mode, 'Usage: node scripts/reading-digest.mjs --write|--check [--root directory]');
  const result = await generate({ root, mode });
  process.stdout.write(`${mode === 'write' ? 'Generated' : 'Verified'} ${result.files} files from ${result.digests} digest(s).\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch(error => {
    process.stderr.write(`Reading digest: ${error.message}\n`);
    process.exitCode = 1;
  });
}
