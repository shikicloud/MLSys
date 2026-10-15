#!/usr/bin/env node
import { chmod, lstat, mkdir, open, readFile, readdir, rename, unlink } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';

const defaultRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const defaultConfigDir = path.join(os.homedir(), '.config', 'mlsys-reading');
const publicRoot = 'https://shikicloud.github.io/MLSys';
const feedUrl = `${publicRoot}/reading.xml`;
const readerUrl = 'netnewswire://';
const safeError = () => new Error('MLSys notification operation could not be completed.');
const requireValue = condition => { if (!condition) throw safeError(); };

function validDate(date) {
  return typeof date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(date)
    && Number.isFinite(Date.parse(`${date}T00:00:00Z`))
    && new Date(`${date}T00:00:00Z`).toISOString().slice(0, 10) === date;
}

const issueUrl = date => `${publicRoot}/CN/reading/daily/${date}`;
const shortText = (value, limit) => {
  requireValue(typeof value === 'string' && value.trim() && !/[\u0000-\u001f\u007f]/.test(value));
  const characters = Array.from(value.trim());
  return characters.length <= limit ? characters.join('') : characters.slice(0, limit - 1).join('') + '…';
};

async function privateDirectory(directory, create = false) {
  if (create) await mkdir(directory, { recursive: true, mode: 0o700 });
  let info;
  try { info = await lstat(directory); } catch (error) {
    if (!create && error.code === 'ENOENT') return false;
    throw safeError();
  }
  requireValue(info.isDirectory() && !info.isSymbolicLink());
  await chmod(directory, 0o700);
  return true;
}

async function readPrivate(directory, name) {
  if (!await privateDirectory(directory)) return undefined;
  const filename = path.join(directory, name);
  let info;
  try { info = await lstat(filename); } catch (error) {
    if (error.code === 'ENOENT') return undefined;
    throw safeError();
  }
  requireValue(info.isFile() && !info.isSymbolicLink());
  await chmod(filename, 0o600);
  try { return JSON.parse(await readFile(filename, 'utf8')); } catch { throw safeError(); }
}

async function writePrivate(directory, name, value) {
  await privateDirectory(directory, true);
  const destination = path.join(directory, name);
  try {
    const info = await lstat(destination);
    requireValue(info.isFile() && !info.isSymbolicLink());
  } catch (error) { if (error.code !== 'ENOENT') throw safeError(); }
  const temporary = path.join(directory, `.${name}.${randomUUID()}.tmp`);
  try {
    const file = await open(temporary, 'wx', 0o600);
    try { await file.writeFile(JSON.stringify(value, null, 2) + '\n'); } finally { await file.close(); }
    await rename(temporary, destination);
    await chmod(destination, 0o600);
  } finally {
    await unlink(temporary).catch(() => {});
  }
}

function validateConfig(config) {
  requireValue(config && typeof config.endpoint === 'string' && typeof config.deviceKey === 'string');
  let endpoint;
  try { endpoint = new URL(config.endpoint); } catch { throw safeError(); }
  requireValue(endpoint.protocol === 'https:' && !endpoint.username && !endpoint.password
    && endpoint.pathname === '/push' && !endpoint.search && !endpoint.hash);
  requireValue(config.deviceKey.length > 0 && config.deviceKey.length <= 512
    && !/[\s/\\\u0000-\u001f\u007f]/.test(config.deviceKey));
  return { endpoint: endpoint.href, deviceKey: config.deviceKey };
}

export async function configure(input, { configDir = defaultConfigDir } = {}) {
  try {
    requireValue(typeof input === 'string' && input.length <= 8192 && !/[\r\n]/.test(input.trim()));
    const url = new URL(input.trim());
    requireValue(url.protocol === 'https:' && !url.username && !url.password && url.hostname);
    const first = url.pathname.split('/')[1];
    requireValue(first);
    const config = validateConfig({ endpoint: `${url.origin}/push`, deviceKey: decodeURIComponent(first) });
    await writePrivate(configDir, 'bark.json', config);
    return { status: 'configured' };
  } catch { throw safeError(); }
}

async function readDelivery(configDir, date) {
  requireValue(validDate(date));
  const delivery = await readPrivate(configDir, `delivery-${date}.json`);
  if (delivery === undefined) return undefined;
  requireValue(delivery && delivery.date === date && delivery.url === issueUrl(date)
    && ['pending', 'accepted', 'ambiguous', 'failed'].includes(delivery.status));
  return delivery;
}

export async function getStatus({ configDir = defaultConfigDir } = {}) {
  const config = await readPrivate(configDir, 'bark.json');
  if (config !== undefined) validateConfig(config);
  const deliveries = [];
  if (await privateDirectory(configDir)) {
    for (const name of (await readdir(configDir)).sort()) {
      const match = name.match(/^delivery-(\d{4}-\d{2}-\d{2})\.json$/);
      if (match) deliveries.push(await readDelivery(configDir, match[1]));
    }
  }
  const dates = statuses => deliveries.filter(d => d && statuses.includes(d.status)).map(d => d.date).sort();
  return { configured: config !== undefined, acceptedDates: dates(['accepted']), uncertainDates: dates(['pending', 'ambiguous']) };
}

async function readIssue(date, root) {
  requireValue(validDate(date));
  const filename = path.join(root, 'reading', 'digests', `${date}.json`);
  const info = await lstat(filename);
  requireValue(info.isFile() && !info.isSymbolicLink());
  const digest = JSON.parse(await readFile(filename, 'utf8'));
  requireValue(digest.schemaVersion === 1 && digest.date === date && Array.isArray(digest.papers));
  const mains = digest.papers.filter(paper => paper?.role === 'main');
  requireValue(mains.length === 1);
  const theme = shortText(digest.zh?.title, 120);
  const main = shortText(mains[0].title, 140);
  return {
    issueTitle: digest.zh.title,
    publicUrl: issueUrl(date),
    payload: {
      title: `📚 MLSys 每日选读 · ${date}`,
      body: `主题：${theme}\n主读：${main}\n点此打开 NetNewsWire，在 MLSys 订阅读取全文。`,
      url: readerUrl, group: 'MLSys 每日选读', isArchive: 1, id: `mlsys-reading-${date}`,
    },
  };
}

export async function preview(date, { root = defaultRoot } = {}) {
  try { return (await readIssue(date, root)).payload; } catch { throw safeError(); }
}

// Only the feed's scalar title and GUID are needed; do not interpret HTML bodies.
function xmlText(value) {
  if (/^<!\[CDATA\[[\s\S]*\]\]>$/.test(value)) return value.slice(9, -3);
  requireValue(!value.includes('<'));
  return value.replace(/&(#x[\da-fA-F]+|#\d+|amp|lt|gt|quot|apos);/g, (_, entity) => {
    if (entity[0] === '#') {
      const code = entity[1] === 'x' ? parseInt(entity.slice(2), 16) : Number(entity.slice(1));
      requireValue(Number.isInteger(code) && code >= 0 && code <= 0x10ffff);
      return String.fromCodePoint(code);
    }
    return { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'" }[entity];
  });
}

async function verifyPublished(issue, fetchImpl) {
  try {
    const response = await fetchImpl(feedUrl, { redirect: 'error', signal: AbortSignal.timeout(20_000) });
    if (response.status !== 200 || response.redirected) return false;
    const xml = await response.text();
    // Remove full-text CDATA so article text cannot masquerade as RSS metadata.
    const feed = xml.replace(/<content:encoded\b[^>]*>[\s\S]*?<\/content:encoded>/g, '');
    for (const match of feed.matchAll(/<item\b[^>]*>([\s\S]*?)<\/item>/g)) {
      const title = match[1].match(/<title\b[^>]*>([\s\S]*?)<\/title>/);
      const guid = match[1].match(/<guid\b[^>]*>([\s\S]*?)<\/guid>/);
      if (title && guid && xmlText(title[1]) === issue.issueTitle && xmlText(guid[1]) === issue.publicUrl) return true;
    }
    return false;
  } catch { return false; }
}

export async function notifyReading(date, { root = defaultRoot, configDir = defaultConfigDir, fetchImpl = fetch } = {}) {
  try {
    const issue = await readIssue(date, root);
    const config = validateConfig(await readPrivate(configDir, 'bark.json'));
    const lockPath = path.join(configDir, `notify-${date}.lock`);
    const deliveryName = `delivery-${date}.json`;
    let lock;
    try { lock = await open(lockPath, 'wx', 0o600); } catch (error) {
      if (error.code === 'EEXIST') return { status: 'busy' };
      throw safeError();
    }
    try {
      await lock.writeFile(JSON.stringify({ pid: process.pid }) + '\n');
      let delivery = await readDelivery(configDir, date);
      if (delivery?.status === 'accepted') return { status: 'accepted', duplicate: true };
      if (delivery && ['pending', 'ambiguous'].includes(delivery.status)) {
        if (delivery.status === 'pending') {
          delivery.status = 'ambiguous';
          delivery.updatedAt = new Date().toISOString();
          await writePrivate(configDir, deliveryName, delivery);
        }
        return { status: 'ambiguous', duplicate: true };
      }
      if (!await verifyPublished(issue, fetchImpl)) return { status: 'unverified' };
      if (!delivery) {
        delivery = { date, url: issue.publicUrl };
      }
      Object.assign(delivery, { status: 'pending', updatedAt: new Date().toISOString() });
      await writePrivate(configDir, deliveryName, delivery);
      let status = 'ambiguous';
      try {
        const response = await fetchImpl(config.endpoint, {
          method: 'POST', redirect: 'error', signal: AbortSignal.timeout(20_000),
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ ...issue.payload, device_key: config.deviceKey }),
        });
        if (response.redirected) status = 'ambiguous';
        else if (response.status >= 400 && response.status < 500 && response.status !== 408) status = 'failed';
        else if (response.status === 200) {
          const body = await response.json();
          if (typeof body?.code === 'number') status = body.code === 200 ? 'accepted' : 'failed';
        }
      } catch { /* A request may have reached Bark. Never automatically resend. */ }
      Object.assign(delivery, { status, updatedAt: new Date().toISOString() });
      await writePrivate(configDir, deliveryName, delivery);
      return { status };
    } finally {
      await lock.close();
      await unlink(lockPath);
    }
  } catch { throw safeError(); }
}

async function inputLine(input) {
  const lines = createInterface({ input, crlfDelay: Infinity, terminal: false });
  try {
    for await (const line of lines) {
      requireValue(line.length <= 8192);
      return line;
    }
    throw safeError();
  } finally { lines.close(); }
}

export async function runCli(args, options = {}) {
  const { input = process.stdin, output = process.stdout, error = process.stderr } = options;
  try {
    const [command, date] = args;
    requireValue(args.length === (['--preview', '--date'].includes(command) ? 2 : 1));
    if (command === '--configure') {
      await configure(await inputLine(input), options);
      output.write('已配置\n');
    } else if (command === '--status') {
      const status = await getStatus(options);
      output.write(`${status.configured ? 'configured' : '未配置'}\n已发送（服务器已受理）：${status.acceptedDates.join(', ') || '无'}\n`);
      if (status.uncertainDates.length) output.write(`发送结果待核对：${status.uncertainDates.join(', ')}\n`);
    } else if (command === '--preview') {
      output.write(JSON.stringify(await preview(date, options), null, 2) + '\n');
    } else if (command === '--date') {
      const result = await notifyReading(date, options);
      const messages = {
        accepted: result.duplicate ? '该期此前已获服务器受理，未重复发送。' : 'Bark 服务器已受理；这不代表手机已显示通知。',
        ambiguous: '发送结果不确定，已停止自动重发，请先核对手机。',
        failed: '服务器明确未受理，可稍后重试。',
        unverified: '尚未确认该期已发布，未发送通知。',
        busy: '检测到该期通知处理锁，未重复发送，请先核对处理状态。',
      };
      output.write(messages[result.status] + '\n');
      return result.status === 'accepted' ? 0 : 1;
    } else throw safeError();
    return 0;
  } catch {
    error.write('操作未完成，请检查本地配置与输入；未输出任何私密配置或服务器详情。\n');
    return 1;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = await runCli(process.argv.slice(2));
}
