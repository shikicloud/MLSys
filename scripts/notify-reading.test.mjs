import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, stat, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import { configure, getStatus, notifyReading, preview, runCli } from './notify-reading.mjs';

const date = '2026-10-15';
const publicUrl = `https://shikicloud.github.io/MLSys/CN/reading/daily/${date}`;
const key = 'test-private-device-key';
const privateUrl = `https://api.day.app/${key}/ignore-title/ignore-body?ttl=600`;
const title = '训练 & rollout <边界>';
const feed = (guid = publicUrl, itemTitle = '训练 &amp; rollout &lt;边界&gt;') =>
  `<rss><channel><item><title>${itemTitle}</title><guid isPermaLink="true">${guid}</guid><content:encoded>&lt;p&gt;Body&lt;/p&gt;</content:encoded></item></channel></rss>`;
const response = (body, status = 200) => new Response(typeof body === 'string' ? body : JSON.stringify(body), { status });

async function fixture(t, withConfig = true) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'mlsys-notify-test-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const configDir = path.join(root, 'private');
  await mkdir(path.join(root, 'reading/digests'), { recursive: true });
  await writeFile(path.join(root, `reading/digests/${date}.json`), JSON.stringify({
    schemaVersion: 1, date, zh: { title }, papers: [{ role: 'main', title: 'HybridFlow: A Flexible and Efficient RLHF Framework' }],
  }));
  if (withConfig) await configure(privateUrl, { configDir });
  return { root, configDir };
}

test('configuration and CLI output keep secrets private and discard optional URL fields', async t => {
  const options = await fixture(t, false);
  assert.deepEqual(await getStatus(options), { configured: false, acceptedDates: [], uncertainDates: [] });
  let output = '';
  const io = { ...options, input: Readable.from([privateUrl + '\n']), output: { write: value => { output += value; } }, error: { write: value => { output += value; } } };
  assert.equal(await runCli(['--configure'], io), 0);
  assert.equal(output, '已配置\n');
  const configPath = path.join(options.configDir, 'bark.json');
  assert.deepEqual(JSON.parse(await readFile(configPath, 'utf8')), { endpoint: 'https://api.day.app/push', deviceKey: key });
  assert.equal((await stat(options.configDir)).mode & 0o777, 0o700);
  assert.equal((await stat(configPath)).mode & 0o777, 0o600);
  await runCli(['--status'], io);
  for (const invalid of ['http://api.day.app/key', 'https://user:pass@api.day.app/key', 'https://api.day.app/', 'not-a-url']) {
    await assert.rejects(configure(invalid, options), /could not be completed/);
  }
  await runCli(['--force', privateUrl], io);
  assert.ok(!output.includes(key));
  assert.ok(!output.includes(privateUrl));
  assert.ok(!output.includes('api.day.app'));
});

test('preview needs no secret and rejects unsafe dates or mismatched JSON', async t => {
  const options = await fixture(t, false);
  const payload = await preview(date, options);
  assert.equal(payload.title, `📚 MLSys 每日选读 · ${date}`);
  assert.ok(payload.body.includes(title));
  assert.ok(payload.body.includes('主读：HybridFlow'));
  assert.equal(payload.url, publicUrl);
  assert.equal(payload.group, 'MLSys 每日选读');
  assert.equal(payload.isArchive, 1);
  assert.ok(!JSON.stringify(payload).includes(key));
  for (const invalid of ['../bark', '2026-02-30', '2026-10-15/other']) await assert.rejects(preview(invalid, options));
  await writeFile(path.join(options.root, `reading/digests/${date}.json`), JSON.stringify({ schemaVersion: 1, date: '2026-10-14' }));
  await assert.rejects(preview(date, options));
});

test('a verified publication is POSTed once with pending persisted before the request', async t => {
  const options = await fixture(t);
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({ url, init });
    assert.equal(init.redirect, 'error');
    assert.ok(init.signal instanceof AbortSignal);
    if (calls.length === 1) {
      assert.equal(url, 'https://shikicloud.github.io/MLSys/reading.xml');
      assert.equal(init.method, undefined);
      return response(feed());
    }
    assert.equal(url, 'https://api.day.app/push');
    assert.equal(init.method, 'POST');
    const delivery = JSON.parse(await readFile(path.join(options.configDir, `delivery-${date}.json`), 'utf8'));
    assert.equal(delivery.status, 'pending');
    assert.deepEqual(JSON.parse(init.body), { ...await preview(date, options), device_key: key });
    return response({ code: 200, message: `server-private-body-${key}` });
  };
  assert.deepEqual(await notifyReading(date, { ...options, fetchImpl }), { status: 'accepted' });
  assert.deepEqual(await notifyReading(date, { ...options, fetchImpl }), { status: 'accepted', duplicate: true });
  assert.equal(calls.length, 2);
  assert.deepEqual(await getStatus(options), { configured: true, acceptedDates: [date], uncertainDates: [] });
  const stateFile = path.join(options.configDir, `delivery-${date}.json`);
  assert.equal((await stat(stateFile)).mode & 0o777, 0o600);
  const stored = await readFile(stateFile, 'utf8');
  assert.ok(!stored.includes(key));
  assert.ok(!stored.includes('server-private-body'));
  await assert.rejects(stat(path.join(options.configDir, `notify-${date}.lock`)), { code: 'ENOENT' });
});

test('publication gate requires the same item GUID and title and refuses feed redirects', async t => {
  for (const published of [
    response(feed(publicUrl, 'Wrong title')),
    response(feed(`${publicUrl}-wrong`)),
    response('<rss><item><title>训练 &amp; rollout &lt;边界&gt;</title><guid>wrong</guid></item><item><title>Wrong</title><guid>' + publicUrl + '</guid></item></rss>'),
    response('redirect', 302),
  ]) {
    const options = await fixture(t);
    let calls = 0;
    const result = await notifyReading(date, { ...options, fetchImpl: async (_, init) => {
      calls++;
      assert.equal(init.redirect, 'error');
      return published;
    } });
    assert.deepEqual(result, { status: 'unverified' });
    assert.equal(calls, 1, 'no POST before live publication is verified');
    await assert.rejects(stat(path.join(options.configDir, `delivery-${date}.json`)), { code: 'ENOENT' });
  }
});

test('explicit HTTP or API rejection can retry after publication is checked again', async t => {
  for (const rejection of [response('private server rejection', 400), response('unauthorized', 401), response({ code: 400, message: key })]) {
    const options = await fixture(t);
    let calls = 0;
    const fetchImpl = async (_, init) => {
      assert.equal(init.redirect, 'error');
      calls++;
      if (!init.method) return response(feed());
      return calls === 2 ? rejection : response({ code: 200 });
    };
    assert.deepEqual(await notifyReading(date, { ...options, fetchImpl }), { status: 'failed' });
    assert.deepEqual(await notifyReading(date, { ...options, fetchImpl }), { status: 'accepted' });
    assert.equal(calls, 4);
  }
});

test('timeouts and malformed success responses become ambiguous and cannot automatically resend', async t => {
  for (const post of [async () => { throw new Error(`timeout private URL ${privateUrl}`); }, async () => response('not-json'), async () => response({ unexpected: 'reply' })]) {
    const options = await fixture(t);
    let calls = 0;
    const fetchImpl = async (_, init) => {
      calls++;
      return init.method ? post() : response(feed());
    };
    let output = '';
    const io = { ...options, fetchImpl, output: { write: value => { output += value; } }, error: { write: value => { output += value; } } };
    assert.equal(await runCli(['--date', date], io), 1);
    assert.deepEqual(await notifyReading(date, { ...options, fetchImpl }), { status: 'ambiguous', duplicate: true });
    assert.equal(calls, 2);
    assert.ok(!output.includes(key));
    assert.ok(!output.includes('not-json'));
    assert.deepEqual((await getStatus(options)).uncertainDates, [date]);
  }
});

test('HTTP 502, 504, 408 and unexpected redirects are ambiguous and never automatically retried', async t => {
  for (const status of [502, 504, 408, 302]) {
    const options = await fixture(t);
    let calls = 0;
    const fetchImpl = async (_, init) => {
      calls++;
      assert.equal(init.redirect, 'error');
      return init.method ? response('private uncertain server response', status) : response(feed());
    };
    assert.deepEqual(await notifyReading(date, { ...options, fetchImpl }), { status: 'ambiguous' });
    assert.deepEqual(await notifyReading(date, { ...options, fetchImpl }), { status: 'ambiguous', duplicate: true });
    assert.equal(calls, 2);
  }
});

test('exclusive locking prevents concurrent duplicate delivery, and abandoned pending state blocks retries', async t => {
  const options = await fixture(t);
  let entered;
  const duringPost = new Promise(resolve => { entered = resolve; });
  let release;
  const finishPost = new Promise(resolve => { release = resolve; });
  let posts = 0;
  const fetchImpl = async (_, init) => {
    if (!init.method) return response(feed());
    posts++;
    entered();
    await finishPost;
    return response({ code: 200 });
  };
  const sending = notifyReading(date, { ...options, fetchImpl });
  await duringPost;
  assert.deepEqual(await notifyReading(date, { ...options, fetchImpl }), { status: 'busy' });
  release();
  assert.deepEqual(await sending, { status: 'accepted' });
  assert.equal(posts, 1);
  await writeFile(path.join(options.configDir, `delivery-${date}.json`), JSON.stringify({ date, url: publicUrl, status: 'pending' }));
  assert.deepEqual(await notifyReading(date, { ...options, fetchImpl: () => { throw new Error('must not fetch'); } }), { status: 'ambiguous', duplicate: true });
});

test('an abandoned lock and pending delivery from an old date cannot block a new issue', async t => {
  const options = await fixture(t);
  const oldDate = '2026-10-14';
  await writeFile(path.join(options.root, `reading/digests/${oldDate}.json`), JSON.stringify({
    schemaVersion: 1, date: oldDate, zh: { title: 'Older issue' }, papers: [{ role: 'main', title: 'Older main read' }],
  }));
  const oldLock = path.join(options.configDir, `notify-${oldDate}.lock`);
  const oldDelivery = path.join(options.configDir, `delivery-${oldDate}.json`);
  await writeFile(oldLock, 'abandoned', { mode: 0o600 });
  await writeFile(oldDelivery, JSON.stringify({ date: oldDate, url: publicUrl.replace(date, oldDate), status: 'pending' }), { mode: 0o600 });
  let posts = 0;
  const fetchImpl = async (_, init) => {
    if (!init.method) return response(feed());
    posts++;
    return response({ code: 200 });
  };
  assert.deepEqual(await notifyReading(date, { ...options, fetchImpl }), { status: 'accepted' });
  assert.deepEqual(await notifyReading(oldDate, { ...options, fetchImpl }), { status: 'busy' });
  assert.equal(posts, 1);
  assert.equal(await readFile(oldLock, 'utf8'), 'abandoned');
  assert.equal(JSON.parse(await readFile(oldDelivery, 'utf8')).status, 'pending');
  assert.deepEqual(await getStatus(options), { configured: true, acceptedDates: [date], uncertainDates: [oldDate] });
});
