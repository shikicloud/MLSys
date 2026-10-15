import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm, stat, symlink } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { generate } from './reading-digest.mjs';

const now = new Date('2026-10-16T12:00:00+08:00');
const fixture = (date = '2026-10-15') => ({
  schemaVersion: 1, date, publishedAt: `${date}T09:00:00+08:00`,
  zh: { title: '从批处理开始', summary: '理解 serving 的基础。' },
  en: { title: 'Start with batching', summary: 'Understand serving foundations.' },
  papers: ['orca-2022', 'vllm-2023', 'sglang-2024'].map((id, i) => ({
    id, title: ['Orca', 'PagedAttention', 'SGLang'][i], year: 2022 + i, venue: 'Verified proceedings',
    url: `https://example.org/papers/${id}`, role: i ? 'optional' : 'main',
    zh: { contribution: '按迭代调度。', relevance: '为 rollout 调度建立基础。', reading: '先读架构图。', prerequisites: '自回归生成。', caveat: '不覆盖所有工具等待。' },
    en: { contribution: 'Iteration scheduling.', relevance: 'A foundation for rollout scheduling.', reading: 'Start with the architecture.', prerequisites: 'Autoregressive generation.', caveat: 'Does not cover all tool waits.' },
  })),
});

const detailedFixture = () => {
  const record = fixture();
  record.papers[0].zh.sections = [
    { heading: '背景与核心设计', paragraphs: ['**迭代调度**从请求长度不一的问题出发。', '每次生成一个 token 后重新决定批次成员。'] },
    { heading: '系统联系、先修与局限', paragraphs: ['以自回归生成知识分析 rollout 调度；工具等待仍需独立策略。'] },
  ];
  record.papers[0].en.sections = [
    { heading: 'Motivation and design', paragraphs: ['**Iteration scheduling** addresses requests of unequal lengths.', 'Reconsider batch membership after each token.'] },
    { heading: 'Practical relevance, prerequisites, and limits', paragraphs: ['Use autoregressive generation to reason about rollout scheduling.', 'Tool waits still require their own policy.'] },
  ];
  return record;
};

async function setup(t, records = [fixture()]) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'mlsys-reading-test-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(path.join(root, 'reading/digests'), { recursive: true });
  for (const record of records) await put(root, record);
  return root;
}

async function put(root, record, filename = `${record.date}.json`) {
  await writeFile(path.join(root, 'reading/digests', filename), JSON.stringify(record));
}

test('bilingual output preserves publication time, explicit recommendation status and stable files', async t => {
  const root = await setup(t);
  assert.deepEqual(await generate({ root, mode: 'write', now }), { digests: 1, files: 4 });
  const cnPath = path.join(root, 'CN/reading/daily/2026-10-15.md');
  const cn = await readFile(cnPath, 'utf8');
  const en = await readFile(path.join(root, 'EN/reading/daily/2026-10-15.md'), 'utf8');
  assert.match(cn, /候选不等于已读/);
  assert.match(en, /does not mean it has been read/);
  assert.match(cn, /published: "2026-10-15T09:00:00\+08:00"/);
  assert.match(cn, /\[\[CN\/reading\/roadmap\|/);
  assert.match(en, /\[\[EN\/reading\/index\|/);
  assert.match(cn, /核心贡献/);
  assert.match(en, /Core contribution/);
  assert.match(cn, /https:\/\/example.org\/papers\/orca-2022/);
  assert.equal((cn.match(/^## /gm) ?? []).length, 3);
  const before = (await stat(cnPath)).mtimeMs;
  await generate({ root, mode: 'write', now: new Date('2026-10-17T00:00:00Z') });
  assert.equal((await stat(cnPath)).mtimeMs, before);
  await generate({ root, mode: 'check', now });
  await writeFile(cnPath, cn + '\nmanual drift');
  await assert.rejects(generate({ root, mode: 'check', now }), /missing or outdated/);
});

test('optional bilingual sections render long-form paragraphs while legacy papers stay unchanged', async t => {
  const record = detailedFixture();
  record.papers[0].revisitReason = { zh: '从调度视角回顾。', en: 'Revisit the scheduler perspective.' };
  const root = await setup(t, [record]);
  await generate({ root, mode: 'write', now });
  const outputs = new Map();
  for (const [lang, folder, main, optional, guidance, reason] of [
    ['zh', 'CN', '主读', '选读', '阅读建议', '再次推荐的原因'],
    ['en', 'EN', 'Main reading', 'Optional reading', 'Reading guidance', 'Why revisit'],
  ]) {
    const destination = path.join(root, folder, 'reading/daily/2026-10-15.md');
    const content = await readFile(destination, 'utf8');
    const mainBody = content.split(`## ${main} · Orca\n\n`)[1].split(`## ${optional} ·`)[0];
    assert.match(mainBody, /2022 · Verified proceedings/);
    assert.match(mainBody, /https:\/\/example.org\/papers\/orca-2022/);
    assert.ok(mainBody.includes(`**${guidance}：** ${record.papers[0][lang].reading}`));
    assert.ok(mainBody.includes(`**${reason}：** ${record.papers[0].revisitReason[lang]}`));
    assert.doesNotMatch(mainBody, /^- /m, 'long-form metadata must not become a checklist');
    assert.ok(!mainBody.includes(record.papers[0][lang].contribution), 'summary text is retained in JSON, not duplicated in the long form');
    for (const section of record.papers[0][lang].sections) {
      assert.ok(mainBody.includes(`### ${section.heading}\n\n${section.paragraphs.join('\n\n')}\n`));
    }
    assert.ok(content.includes(record.papers[1][lang].contribution), 'legacy optional papers retain their summaries');
    assert.match(content, /published: "2026-10-15T09:00:00\+08:00"/);
    outputs.set(destination, { content, mtime: (await stat(destination)).mtimeMs });
  }
  await generate({ root, mode: 'write', now: new Date('2026-10-17T00:00:00Z') });
  await generate({ root, mode: 'check', now });
  for (const [destination, previous] of outputs) {
    assert.equal(await readFile(destination, 'utf8'), previous.content);
    assert.equal((await stat(destination)).mtimeMs, previous.mtime);
  }
});

test('block syntax and open fences cannot consume later sections or papers', async t => {
  for (const paragraph of [
    '```', '```js', '   ````python', '~~~', '~~~inline~~~',
    '# Replaces a heading', '   ### Nested heading', '> quote', '>without a space',
    '- list', '+ list', '* list', '1. numbered list', '2) numbered list',
    '    indented code', '---', '* * *', '___',
    '[source]: https://example.org', '[^note]: footnote', '$$', '$$$ E=mc^2',
  ]) {
    const record = detailedFixture();
    record.papers[0].zh.sections[0].paragraphs = [paragraph];
    const root = await setup(t, [record]);
    for (const mode of ['write', 'check']) {
      await assert.rejects(generate({ root, mode, now }), /inline Markdown only, not block syntax/, paragraph);
    }
    await assert.rejects(stat(path.join(root, 'CN')), { code: 'ENOENT' });
    await assert.rejects(stat(path.join(root, 'EN')), { code: 'ENOENT' });
  }
});

test('section paragraphs preserve inline links, math, emphasis, and code without false block matches', async t => {
  const paragraphs = [
    '[Original source](https://example.org/paper?x=1&y=2#results)',
    '**Main result.** Read *this comparison* first.',
    '`cache.match()` returns a boundary.', '```inline code containing `backticks` ```',
    '$x < y$ and $a > b$ are inline comparisons.', '$$E=mc^2$$',
    '-1 is negative; +1 is positive.', '1.5× throughput is not an ordered list.',
    'a * b - c is arithmetic.', '#hashtag is text.', '\\# A literal heading marker.',
  ];
  const record = detailedFixture();
  for (const lang of ['zh', 'en']) record.papers[0][lang].sections[0].paragraphs = paragraphs;
  const root = await setup(t, [record]);
  await generate({ root, mode: 'write', now });
  await generate({ root, mode: 'check', now });
  for (const folder of ['CN', 'EN']) {
    const content = await readFile(path.join(root, folder, 'reading/daily/2026-10-15.md'), 'utf8');
    assert.ok(content.includes(paragraphs.join('\n\n')));
    assert.match(content, /## (?:选读|Optional reading) · PagedAttention/);
  }
});

const invalidSections = [
  ['missing English sections', p => { delete p.en.sections; }, /sections must be arrays in both zh and en/],
  ['missing Chinese sections', p => { delete p.zh.sections; }, /sections must be arrays in both zh and en/],
  ['non-array sections', p => { p.zh.sections = {}; }, /sections must be arrays/],
  ['null sections', p => { p.zh.sections = null; }, /sections must be arrays/],
  ['empty sections', p => { p.zh.sections = []; p.en.sections = []; }, /sections must be nonempty/],
  ['mismatched section counts', p => { p.en.sections.pop(); }, /matching zh and en section counts/],
  ['non-object section', p => { p.zh.sections[0] = 'paragraph'; }, /sections\[0\] must be an object/],
  ['null section', p => { p.zh.sections[0] = null; }, /sections\[0\] must be an object/],
  ['missing heading', p => { delete p.zh.sections[0].heading; }, /heading must be nonempty text/],
  ['blank heading', p => { p.zh.sections[0].heading = ' '; }, /heading must be nonempty text/],
  ['HTML heading', p => { p.zh.sections[0].heading = '<b>Design</b>'; }, /heading must not contain HTML/],
  ['non-array paragraphs', p => { p.zh.sections[0].paragraphs = 'text'; }, /paragraphs must be a nonempty array/],
  ['empty paragraphs', p => { p.zh.sections[0].paragraphs = []; }, /paragraphs must be a nonempty array/],
  ['non-text paragraph', p => { p.zh.sections[0].paragraphs[0] = { text: 'text' }; }, /paragraphs\[0\] must be nonempty text/],
  ['blank paragraph', p => { p.zh.sections[0].paragraphs[0] = ' '; }, /paragraphs\[0\] must be nonempty text/],
  ['HTML paragraph', p => { p.zh.sections[0].paragraphs[0] = '<script>bad</script>'; }, /paragraphs\[0\] must not contain HTML/],
  ['control characters in a paragraph', p => { p.zh.sections[0].paragraphs[0] = 'first\nsecond'; }, /without control characters/],
  ['missing summary alongside sections', p => { delete p.en.caveat; }, /en.caveat must be nonempty text/],
];

for (const [name, mutate, error] of invalidSections) {
  test(`rejects ${name} before writing any long-form output`, async t => {
    const record = detailedFixture();
    mutate(record.papers[0]);
    const root = await setup(t, [record]);
    for (const mode of ['write', 'check']) await assert.rejects(generate({ root, mode, now }), error);
    await assert.rejects(stat(path.join(root, 'CN')), { code: 'ENOENT' });
    await assert.rejects(stat(path.join(root, 'EN')), { code: 'ENOENT' });
  });
}

test('repeated IDs require a bilingual revisit reason and archive sorts newest first', async t => {
  const first = fixture();
  const second = fixture('2026-10-16');
  const root = await setup(t, [first, second]);
  await assert.rejects(generate({ root, mode: 'write', now }), /revisitReason is required/);
  await assert.rejects(stat(path.join(root, 'CN')), { code: 'ENOENT' });
  for (const p of second.papers) p.revisitReason = { zh: '从调度视角重新阅读。', en: 'Revisit from the scheduler perspective.' };
  await put(root, second);
  await generate({ root, mode: 'write', now });
  const index = await readFile(path.join(root, 'CN/reading/index.md'), 'utf8');
  assert.ok(index.indexOf('daily/2026-10-16') < index.indexOf('daily/2026-10-15'));
  const day = await readFile(path.join(root, 'EN/reading/daily/2026-10-16.md'), 'utf8');
  assert.match(day, /Why revisit.*Revisit from the scheduler perspective/);
  delete second.papers[0].revisitReason.en;
  await put(root, second);
  await assert.rejects(generate({ root, mode: 'write', now }), /revisitReason.en/);
});

test('daily pages and indexes use an absolute feed URL so Quartz cannot repeat the base path', async t => {
  const root = await setup(t);
  await generate({ root, mode: 'write', now });
  for (const folder of ['CN', 'EN']) {
    for (const relative of ['index.md', 'daily/2026-10-15.md']) {
      const content = await readFile(path.join(root, folder, 'reading', relative), 'utf8');
      assert.match(content, /\]\(https:\/\/shikicloud\.github\.io\/MLSys\/reading\.xml\)/);
      assert.doesNotMatch(content, /\]\(\/MLSys\/reading\.xml\)/);
    }
  }
});

test('deleted digest sources fail write and check with both orphan paths and leave all pages untouched', async t => {
  const first = fixture();
  const second = fixture('2026-10-16');
  for (const p of second.papers) {
    p.id += '-new';
    p.url += '-new';
  }
  const root = await setup(t, [first, second]);
  await generate({ root, mode: 'write', now });
  const before = new Map();
  for (const folder of ['CN', 'EN']) {
    for (const relative of ['index.md', 'daily/2026-10-15.md', 'daily/2026-10-16.md']) {
      const destination = path.join(root, folder, 'reading', relative);
      before.set(destination, { content: await readFile(destination, 'utf8'), mtime: (await stat(destination)).mtimeMs });
    }
  }
  await rm(path.join(root, 'reading/digests/2026-10-15.json'));
  second.zh.summary = 'This valid source change must not produce a partial write.';
  await put(root, second);
  for (const mode of ['write', 'check']) {
    await assert.rejects(generate({ root, mode, now }), error => {
      assert.match(error.message, /no matching digest JSON/);
      for (const folder of ['CN', 'EN']) assert.ok(error.message.includes(path.join(root, folder, 'reading/daily/2026-10-15.md')));
      return true;
    });
    for (const [destination, original] of before) {
      assert.equal(await readFile(destination, 'utf8'), original.content);
      assert.equal((await stat(destination)).mtimeMs, original.mtime);
    }
  }
  await rm(path.join(root, 'reading/digests/2026-10-16.json'));
  await assert.rejects(generate({ root, mode: 'write', now }), /no matching digest JSON/);
});

test('changing IDs cannot bypass cross-date canonical source URL checks; explicit revisits are allowed', async t => {
  const first = fixture();
  const second = fixture('2026-10-16');
  for (const p of second.papers) {
    p.id += '-renamed';
    p.url = p.url.replace('example.org', 'EXAMPLE.ORG:443') + '#evaluation';
  }
  const root = await setup(t, [first, second]);
  for (const mode of ['write', 'check']) {
    await assert.rejects(generate({ root, mode, now }), /source URL was already recommended.*revisitReason is required/);
  }
  await assert.rejects(stat(path.join(root, 'CN')), { code: 'ENOENT' });
  for (const p of second.papers) p.revisitReason = { zh: '补读评测部分。', en: 'Return to the evaluation section.' };
  await put(root, second);
  await generate({ root, mode: 'write', now });
  await generate({ root, mode: 'check', now });
  assert.match(await readFile(path.join(root, 'EN/reading/daily/2026-10-16.md'), 'utf8'), /Why revisit.*Return to the evaluation section/);
});

const invalid = [
  ['impossible calendar date', d => { d.date = '2026-02-30'; d.publishedAt = `${d.date}T09:00:00+08:00`; }, /invalid date/],
  ['mismatched publication date', d => { d.publishedAt = '2026-10-14T23:59:00+08:00'; }, /Asia\/Shanghai/],
  ['wrong timezone', d => { d.publishedAt = '2026-10-15T01:00:00Z'; }, /\+08:00/],
  ['invalid time', d => { d.publishedAt = '2026-10-15T25:00:00+08:00'; }, /ISO time/],
  ['future publication', d => { d.date = '2026-10-17'; d.publishedAt = `${d.date}T09:00:00+08:00`; }, /future publication/],
  ['HTTP link', d => { d.papers[0].url = 'http://example.org'; }, /HTTPS/],
  ['relative link', d => { d.papers[0].url = '/paper.pdf'; }, /HTTPS/],
  ['credential link', d => { d.papers[0].url = 'https://user:secret@example.org'; }, /credentials/],
  ['malformed link', d => { d.papers[0].url = 'https://'; }, /invalid URL/],
  ['unsafe ID', d => { d.papers[0].id = '../outside'; }, /unsafe paper ID/],
  ['duplicate same-day ID', d => { d.papers[1].id = d.papers[0].id; }, /duplicate ID/],
  ['duplicate same-day canonical source URL with a different ID', d => {
    d.papers[1].url = d.papers[0].url.replace('example.org', 'EXAMPLE.ORG:443') + '#details';
    d.papers[1].revisitReason = { zh: '重复条目仍然无效。', en: 'A revisit reason cannot allow a same-day duplicate.' };
  }, /duplicate source URL within a digest/],
  ['too few papers', d => { d.papers.pop(); }, /3–5/],
  ['two main papers', d => { d.papers[1].role = 'main'; }, /exactly one/],
  ['missing translation', d => { delete d.papers[0].en.relevance; }, /nonempty/],
  ['HTML content', d => { d.zh.summary = '<script>alert(1)</script>'; }, /HTML/],
  ['invalid year', d => { d.papers[0].year = 2027; }, /publication year/],
];

for (const [name, mutate, error] of invalid) {
  test(`rejects ${name} before writing outputs`, async t => {
    const record = fixture();
    mutate(record);
    const root = await setup(t, [record]);
    await assert.rejects(generate({ root, mode: 'write', now }), error);
    await assert.rejects(stat(path.join(root, 'CN')), { code: 'ENOENT' });
  });
}

test('rejects mismatched filenames, malformed JSON, missing and empty inputs', async t => {
  const root = await setup(t, []);
  await assert.rejects(generate({ root, mode: 'write', now }), /No digest JSON/);
  await put(root, fixture(), '2026-10-14.json');
  await assert.rejects(generate({ root, mode: 'write', now }), /does not match filename/);
  await writeFile(path.join(root, 'reading/digests/2026-10-14.json'), '{');
  await assert.rejects(generate({ root, mode: 'write', now }), /2026-10-14.json/);
  await rm(path.join(root, 'reading/digests'), { recursive: true });
  await assert.rejects(generate({ root, mode: 'write', now }), { code: 'ENOENT' });
});

test('rejects source and destination symlinks without partial writes', async t => {
  const root = await setup(t);
  const outside = path.join(root, 'outside');
  await mkdir(outside);
  await symlink(outside, path.join(root, 'EN'));
  await assert.rejects(generate({ root, mode: 'write', now }), /Unsafe symlink/);
  await assert.rejects(stat(path.join(root, 'CN')), { code: 'ENOENT' });
  await rm(path.join(root, 'EN'));
  const source = path.join(root, 'reading/digests/2026-10-15.json');
  await rm(source);
  await writeFile(path.join(outside, 'data.json'), JSON.stringify(fixture()));
  await symlink(path.join(outside, 'data.json'), source);
  await assert.rejects(generate({ root, mode: 'write', now }), /Unsafe symlink/);
});

test('CLI fails nonzero for malformed input and unsupported flags', async t => {
  const root = await setup(t, []);
  const script = fileURLToPath(new URL('./reading-digest.mjs', import.meta.url));
  for (const args of [['--check', '--root', root], ['--now', '2030-01-01'], ['--write', '--check']]) {
    const result = spawnSync(process.execPath, [script, ...args], { encoding: 'utf8' });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /Reading digest:/);
  }
});
