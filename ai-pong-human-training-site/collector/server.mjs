import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { appendFile, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { dirname, extname, isAbsolute, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const collectorDir = fileURLToPath(new URL(".", import.meta.url));
const siteRoot = resolve(collectorDir, "..");
loadDotEnvFile(join(siteRoot, ".env"));

const webRoot = resolve(siteRoot, "web");
const dataRoot = resolveDataRoot(process.env.AIPONG_DATA_DIR);
const port = Number(process.env.PORT || process.env.AIPONG_PORT || 8787);
const allowedOrigin = process.env.AIPONG_ALLOWED_ORIGIN || "*";
const maxUploadBytes = Number(process.env.AIPONG_MAX_UPLOAD_BYTES || 150_000);
const storageMode = (process.env.AIPONG_STORAGE || "local").toLowerCase();
const leaderboardLimit = Number(process.env.AIPONG_LEADERBOARD_LIMIT || 25);
const leaderboardFile = join(dataRoot, "leaderboard.json");
const playStatsFile = join(dataRoot, "play-stats.json");
const usageFile = join(dataRoot, "usage-budget.json");
const maxSamplesPerUpload = Number(process.env.AIPONG_MAX_SAMPLES_PER_UPLOAD || 700);
const minTrajectorySamples = Number(process.env.AIPONG_MIN_TRAJECTORY_SAMPLES || 120);
const minTrajectoryDurationMs = Number(process.env.AIPONG_MIN_TRAJECTORY_DURATION_MS || 20_000);
const minInputChanges = Number(process.env.AIPONG_MIN_INPUT_CHANGES || 4);
const minHumanHits = Number(process.env.AIPONG_MIN_HUMAN_HITS || 1);
const minVisibleRatio = Number(process.env.AIPONG_MIN_VISIBLE_RATIO || 0.8);
const maxPongScore = Number(process.env.AIPONG_MAX_PONG_SCORE || 7);
const leaderboardMinDurationMs = Number(process.env.AIPONG_LEADERBOARD_MIN_DURATION_MS || 5_000);
const rateLimitWindowMs = Number(process.env.AIPONG_RATE_LIMIT_WINDOW_MS || 60_000);
const trajectoryRateLimit = Number(process.env.AIPONG_TRAJECTORY_RATE_LIMIT || 30);
const leaderboardRateLimit = Number(process.env.AIPONG_LEADERBOARD_RATE_LIMIT || 12);
const statsRateLimit = Number(process.env.AIPONG_STATS_RATE_LIMIT || 30);
const exposeHealthDetails = process.env.AIPONG_EXPOSE_HEALTH_DETAILS === "1";
const trustProxyHeaders = process.env.AIPONG_TRUST_PROXY_HEADERS === "1";
const usageBudgetUnits = Math.max(0, Number(process.env.AIPONG_USAGE_BUDGET_UNITS || 0));
const usageWindow = String(process.env.AIPONG_USAGE_WINDOW || "monthly").toLowerCase();
const usageSlowdownRatio = clampRatio(process.env.AIPONG_USAGE_SLOWDOWN_RATIO, 0.65);
const usageMaintenanceRatio = clampRatio(process.env.AIPONG_USAGE_MAINTENANCE_RATIO, 0.8, usageSlowdownRatio);
const usageBytesPerUnit = Math.max(1, Number(process.env.AIPONG_USAGE_BYTES_PER_UNIT || 25_000));
const usageTrajectoryBaseUnits = Math.max(0, Number(process.env.AIPONG_USAGE_TRAJECTORY_BASE_UNITS || 1));
const usagePlayUnits = Math.max(0, Number(process.env.AIPONG_USAGE_PLAY_UNITS || 1));
const usageLeaderboardUnits = Math.max(0, Number(process.env.AIPONG_USAGE_LEADERBOARD_UNITS || 1));
const usageMinSlowdownMs = Math.max(0, Number(process.env.AIPONG_USAGE_MIN_SLOWDOWN_MS || 250));
const usageMaxSlowdownMs = Math.max(usageMinSlowdownMs, Number(process.env.AIPONG_USAGE_MAX_SLOWDOWN_MS || 2_500));
const usageRetryAfterSeconds = Math.max(60, Number(process.env.AIPONG_USAGE_RETRY_AFTER_SECONDS || 900));
const pointsPerHit = 10;
const pointsPerRally = 2;
const pointsPerPoint = 100;
let r2StoragePromise = null;
let leaderboardWriteQueue = Promise.resolve();
let playStatsWriteQueue = Promise.resolve();
let usageWriteQueue = Promise.resolve();
const rateLimitBuckets = new Map();
const recentPlayMatchIds = new Map();

const mimeTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".ico", "image/x-icon"],
  [".svg", "image/svg+xml"],
]);

function loadDotEnvFile(filePath) {
  if (!existsSync(filePath)) {
    return;
  }

  const content = readFileSync(filePath, "utf8");
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }

    const match = line.match(/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
    if (!match) {
      continue;
    }

    const [, key, rawValue] = match;
    if (process.env[key] !== undefined) {
      continue;
    }

    let value = rawValue.trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    process.env[key] = value;
  }
}

function resolveDataRoot(configuredPath) {
  if (!configuredPath) {
    return resolve(siteRoot, "data");
  }
  return isAbsolute(configuredPath) ? resolve(configuredPath) : resolve(siteRoot, configuredPath);
}

function writeCorsHeaders(response) {
  response.setHeader("Access-Control-Allow-Origin", allowedOrigin);
  response.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  response.setHeader("Access-Control-Allow-Headers", "Content-Type,X-Aipong-Match-Id,X-Aipong-Chunk-Index,X-Aipong-Reason");
}

function writeSecurityHeaders(response) {
  response.setHeader("X-Content-Type-Options", "nosniff");
  response.setHeader("X-Frame-Options", "DENY");
  response.setHeader("Referrer-Policy", "no-referrer");
  response.setHeader("Cross-Origin-Resource-Policy", "same-origin");
  response.setHeader(
    "Content-Security-Policy",
    "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
  );
}

function sendJson(response, statusCode, payload) {
  writeCorsHeaders(response);
  writeSecurityHeaders(response);
  response.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  response.end(JSON.stringify(payload));
}

function sendRateLimit(response, retryAfterSeconds) {
  writeCorsHeaders(response);
  writeSecurityHeaders(response);
  response.writeHead(429, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "Retry-After": String(retryAfterSeconds),
  });
  response.end(JSON.stringify({ ok: false, error: "Too many requests" }));
}

function sendMaintenanceJson(response, usage) {
  writeCorsHeaders(response);
  writeSecurityHeaders(response);
  response.writeHead(503, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "Retry-After": String(usageRetryAfterSeconds),
  });
  response.end(JSON.stringify({
    ok: false,
    mode: "maintenance",
    error: usage.message,
    usage,
  }));
}

function sanitize(value, fallback = "unknown") {
  const text = String(value || fallback).slice(0, 96);
  const safe = text.replace(/[^a-zA-Z0-9_.-]/g, "-");
  return safe || fallback;
}

function sanitizePlayerName(value) {
  const text = String(value || "").trim().replace(/\s+/g, " ").slice(0, 20);
  const safe = text.replace(/[^\w .-]/g, "");
  return safe || "Anonymous";
}

function clampNumber(value, min, max) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return min;
  }
  return Math.max(min, Math.min(max, number));
}

function clampRatio(value, fallback, floor = 0) {
  const ratio = Number(value);
  if (!Number.isFinite(ratio)) {
    return fallback;
  }
  return Math.max(floor, Math.min(1, ratio));
}

function sleep(ms) {
  if (!ms || ms <= 0) {
    return Promise.resolve();
  }
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

function getClientIp(request) {
  if (trustProxyHeaders) {
    const cloudflareIp = request.headers["cf-connecting-ip"];
    if (typeof cloudflareIp === "string" && cloudflareIp.trim()) {
      return cloudflareIp.trim();
    }

    const forwardedFor = request.headers["x-forwarded-for"];
    if (typeof forwardedFor === "string" && forwardedFor.trim()) {
      return forwardedFor.split(",")[0].trim();
    }
  }
  return request.socket.remoteAddress || "unknown";
}

function checkRateLimit(request, bucketName, limit) {
  if (limit <= 0) {
    return { ok: true, retryAfterSeconds: 0 };
  }

  const now = Date.now();
  const key = `${bucketName}:${getClientIp(request)}`;
  const bucket = rateLimitBuckets.get(key);
  if (!bucket || bucket.resetAt <= now) {
    rateLimitBuckets.set(key, { count: 1, resetAt: now + rateLimitWindowMs });
    return { ok: true, retryAfterSeconds: 0 };
  }

  bucket.count += 1;
  if (bucket.count <= limit) {
    return { ok: true, retryAfterSeconds: 0 };
  }

  return {
    ok: false,
    retryAfterSeconds: Math.max(1, Math.ceil((bucket.resetAt - now) / 1000)),
  };
}

function pruneSmallMaps() {
  const now = Date.now();
  for (const [key, bucket] of rateLimitBuckets) {
    if (bucket.resetAt <= now) {
      rateLimitBuckets.delete(key);
    }
  }
  for (const [key, expiresAt] of recentPlayMatchIds) {
    if (expiresAt <= now) {
      recentPlayMatchIds.delete(key);
    }
  }
}

function normalizeUsageWindow(value) {
  return ["daily", "monthly", "never"].includes(value) ? value : "monthly";
}

function getUsageWindowKey(now = new Date()) {
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  const normalizedWindow = normalizeUsageWindow(usageWindow);
  if (normalizedWindow === "daily") {
    return `${year}-${month}-${day}`;
  }
  if (normalizedWindow === "never") {
    return "lifetime";
  }
  return `${year}-${month}`;
}

function createEmptyUsageStats(now = new Date()) {
  return {
    window: normalizeUsageWindow(usageWindow),
    windowKey: getUsageWindowKey(now),
    updatedAt: now.toISOString(),
    usedUnits: 0,
    trajectoryUploads: 0,
    trajectoryBytes: 0,
    leaderboardPosts: 0,
    playRegistrations: 0,
  };
}

function normalizeUsageStats(parsed) {
  const now = new Date();
  const next = createEmptyUsageStats(now);
  const parsedWindow = normalizeUsageWindow(String(parsed?.window || usageWindow).toLowerCase());
  const parsedWindowKey = String(parsed?.windowKey || "");
  if (parsedWindow !== normalizeUsageWindow(usageWindow) || parsedWindowKey !== getUsageWindowKey(now)) {
    return next;
  }

  next.updatedAt = parsed?.updatedAt || next.updatedAt;
  next.usedUnits = clampNumber(parsed?.usedUnits, 0, Number.MAX_SAFE_INTEGER);
  next.trajectoryUploads = clampNumber(parsed?.trajectoryUploads, 0, Number.MAX_SAFE_INTEGER);
  next.trajectoryBytes = clampNumber(parsed?.trajectoryBytes, 0, Number.MAX_SAFE_INTEGER);
  next.leaderboardPosts = clampNumber(parsed?.leaderboardPosts, 0, Number.MAX_SAFE_INTEGER);
  next.playRegistrations = clampNumber(parsed?.playRegistrations, 0, Number.MAX_SAFE_INTEGER);
  return next;
}

async function readUsageStats() {
  const parsed = usageBudgetUnits > 0
    ? JSON.parse(await readFile(usageFile, "utf8").catch((error) => {
        if (error?.code === "ENOENT") {
          return "{}";
        }
        throw error;
      }))
    : {};
  return normalizeUsageStats(parsed);
}

function describeUsageMode(mode) {
  if (mode === "maintenance") {
    return "The arcade hit its current usage cap and is cooling down. Come back after the next reset.";
  }
  if (mode === "slow") {
    return "The arcade is heating up. Saving matches may feel slower for a bit.";
  }
  return "The arcade is ready for new matches.";
}

function getUsageSnapshot(stats) {
  const usedUnits = clampNumber(stats?.usedUnits, 0, Number.MAX_SAFE_INTEGER);
  if (usageBudgetUnits <= 0) {
    return {
      enabled: false,
      window: normalizeUsageWindow(usageWindow),
      windowKey: getUsageWindowKey(),
      budgetUnits: 0,
      usedUnits,
      ratio: 0,
      percentage: 0,
      mode: "normal",
      slowdownMs: 0,
      message: describeUsageMode("normal"),
      updatedAt: stats?.updatedAt || new Date().toISOString(),
      trajectoryUploads: clampNumber(stats?.trajectoryUploads, 0, Number.MAX_SAFE_INTEGER),
      trajectoryBytes: clampNumber(stats?.trajectoryBytes, 0, Number.MAX_SAFE_INTEGER),
      leaderboardPosts: clampNumber(stats?.leaderboardPosts, 0, Number.MAX_SAFE_INTEGER),
      playRegistrations: clampNumber(stats?.playRegistrations, 0, Number.MAX_SAFE_INTEGER),
    };
  }

  const ratio = Math.max(0, usedUnits / usageBudgetUnits);
  let mode = "normal";
  if (ratio >= usageMaintenanceRatio) {
    mode = "maintenance";
  } else if (ratio >= usageSlowdownRatio) {
    mode = "slow";
  }

  let slowdownMs = 0;
  if (mode === "slow") {
    const progress = usageMaintenanceRatio > usageSlowdownRatio
      ? (ratio - usageSlowdownRatio) / (usageMaintenanceRatio - usageSlowdownRatio)
      : 1;
    slowdownMs = Math.round(
      usageMinSlowdownMs + clampNumber(progress, 0, 1) * (usageMaxSlowdownMs - usageMinSlowdownMs)
    );
  }

  return {
    enabled: true,
    window: normalizeUsageWindow(usageWindow),
    windowKey: stats.windowKey,
    budgetUnits: usageBudgetUnits,
    usedUnits,
    ratio,
    percentage: Math.round(ratio * 1000) / 10,
    mode,
    slowdownMs,
    message: describeUsageMode(mode),
    updatedAt: stats.updatedAt,
    trajectoryUploads: stats.trajectoryUploads,
    trajectoryBytes: stats.trajectoryBytes,
    leaderboardPosts: stats.leaderboardPosts,
    playRegistrations: stats.playRegistrations,
  };
}

async function writeUsageStats(stats) {
  if (usageBudgetUnits <= 0) {
    return;
  }
  await mkdir(dataRoot, { recursive: true });
  await writeFile(usageFile, `${JSON.stringify(stats, null, 2)}\n`, "utf8");
}

async function addUsageUnits(kind, bytes = 0) {
  if (usageBudgetUnits <= 0) {
    return getUsageSnapshot(await readUsageStats());
  }

  usageWriteQueue = usageWriteQueue.then(async () => {
    const stats = await readUsageStats();
    const snapshot = getUsageSnapshot(stats);
    if (snapshot.mode === "maintenance") {
      return stats;
    }

    let unitsAdded = 0;
    if (kind === "trajectory") {
      unitsAdded = usageTrajectoryBaseUnits + Math.ceil(Math.max(0, bytes) / usageBytesPerUnit);
      stats.trajectoryUploads += 1;
      stats.trajectoryBytes += Math.max(0, bytes);
    } else if (kind === "leaderboard") {
      unitsAdded = usageLeaderboardUnits;
      stats.leaderboardPosts += 1;
    } else if (kind === "play") {
      unitsAdded = usagePlayUnits;
      stats.playRegistrations += 1;
    }

    stats.usedUnits += Math.max(0, unitsAdded);
    stats.updatedAt = new Date().toISOString();
    await writeUsageStats(stats);
    return stats;
  });

  return getUsageSnapshot(await usageWriteQueue);
}

async function getCurrentUsageSnapshot() {
  return getUsageSnapshot(await readUsageStats());
}

async function applyUsageSlowdown(usage) {
  if (usage.mode === "slow" && usage.slowdownMs > 0) {
    await sleep(usage.slowdownMs);
  }
}

function cleanObjectKeyPart(value) {
  return String(value || "")
    .replaceAll("\\", "/")
    .replace(/\.\./g, "")
    .replace(/^\/+|\/+$/g, "");
}

function joinObjectKey(...parts) {
  return parts.map(cleanObjectKeyPart).filter(Boolean).join("/");
}

function readBody(request) {
  return new Promise((resolveBody, rejectBody) => {
    const chunks = [];
    let size = 0;

    request.on("data", (chunk) => {
      size += chunk.length;
      if (size > maxUploadBytes) {
        request.destroy();
        rejectBody(new Error(`Upload exceeded ${maxUploadBytes} bytes`));
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => resolveBody(Buffer.concat(chunks)));
    request.on("error", rejectBody);
  });
}

function parseTrajectoryHeader(buffer) {
  if (buffer.length < 64) {
    throw new Error("Body is smaller than the 64-byte trajectory header");
  }

  const magic = buffer.subarray(0, 8).toString("ascii");
  if (magic !== "AIPONG1\0") {
    throw new Error("Invalid AIPONG magic header");
  }

  const version = buffer.readUInt16LE(8);
  const headerBytes = buffer.readUInt16LE(10);
  const recordBytes = buffer.readUInt16LE(12);
  const sampleHz = buffer.readUInt16LE(14);
  const sampleCount = buffer.readUInt32LE(16);
  const checkpointUpdate = buffer.readUInt32LE(20);
  const durationMs = buffer.readUInt32LE(24);
  const finalLeftScore = buffer.readUInt16LE(28);
  const finalRightScore = buffer.readUInt16LE(30);
  const humanHits = buffer.readUInt16LE(32);
  const botHits = buffer.readUInt16LE(34);
  const maxRally = buffer.readUInt16LE(36);
  const inputChanges = buffer.readUInt16LE(38);
  const visibleSamples = buffer.readUInt16LE(40);
  const flags = buffer.readUInt16LE(42);
  const chunkIndex = buffer.readUInt32LE(44);

  if (version !== 1) {
    throw new Error(`Unsupported trajectory version ${version}`);
  }
  if (headerBytes !== 64) {
    throw new Error(`Unexpected header size ${headerBytes}`);
  }
  if (recordBytes !== 48) {
    throw new Error(`Unexpected record size ${recordBytes}`);
  }
  if (sampleHz !== 10) {
    throw new Error(`Unexpected sample rate ${sampleHz}`);
  }
  if (sampleCount < 1 || sampleCount > maxSamplesPerUpload) {
    throw new Error(`Unreasonable sample count ${sampleCount}`);
  }

  const expectedBytes = headerBytes + sampleCount * recordBytes;
  if (buffer.length !== expectedBytes) {
    throw new Error(`Expected ${expectedBytes} bytes, received ${buffer.length}`);
  }

  return {
    version,
    headerBytes,
    recordBytes,
    sampleHz,
    sampleCount,
    checkpointUpdate,
    durationMs,
    finalLeftScore,
    finalRightScore,
    humanHits,
    botHits,
    maxRally,
    inputChanges,
    visibleSamples,
    flags,
    chunkIndex,
  };
}

function analyzeTrajectoryRecords(buffer, header) {
  let visibleSamples = 0;
  let humanHits = 0;
  let botHits = 0;
  let inputChanges = 0;
  let maxRally = 0;
  let lastHumanAction = null;
  let lastLeftScore = 0;
  let lastRightScore = 0;

  for (let index = 0; index < header.sampleCount; index += 1) {
    const offset = header.headerBytes + index * header.recordBytes;
    const humanAction = buffer.readUInt8(offset + 36);
    const botAction = buffer.readUInt8(offset + 37);
    const flags = buffer.readUInt8(offset + 42);
    const leftScore = buffer.readUInt8(offset + 43);
    const rightScore = buffer.readUInt8(offset + 44);
    const rallyHits = buffer.readUInt16LE(offset + 45);
    const visible = buffer.readUInt8(offset + 47);

    if (humanAction > 2 || botAction > 2) {
      throw new Error("Trajectory contains invalid paddle action values");
    }
    if (leftScore > maxPongScore || rightScore > maxPongScore) {
      throw new Error("Trajectory contains an impossible game score");
    }
    if (visible > 1) {
      throw new Error("Trajectory contains an invalid visibility flag");
    }

    if (lastHumanAction !== null && humanAction !== lastHumanAction) {
      inputChanges += 1;
    }
    lastHumanAction = humanAction;
    if (visible === 1) {
      visibleSamples += 1;
    }
    if (flags & 2) {
      humanHits += 1;
    }
    if (flags & 4) {
      botHits += 1;
    }

    maxRally = Math.max(maxRally, rallyHits);
    lastLeftScore = leftScore;
    lastRightScore = rightScore;
  }

  return {
    visibleSamples,
    humanHits,
    botHits,
    inputChanges,
    maxRally,
    lastLeftScore,
    lastRightScore,
  };
}

function validateTrajectoryQuality(header, analysis, reason) {
  const visibleRatio = header.sampleCount > 0 ? analysis.visibleSamples / header.sampleCount : 0;

  if (header.sampleCount < minTrajectorySamples) {
    throw new Error(`Trajectory rejected: too few samples (${header.sampleCount})`);
  }
  if (reason !== "minute" && header.durationMs < minTrajectoryDurationMs) {
    throw new Error("Trajectory rejected: match segment was too short");
  }
  if (analysis.inputChanges < minInputChanges) {
    throw new Error("Trajectory rejected: too little human input variation");
  }
  if (analysis.humanHits < minHumanHits) {
    throw new Error("Trajectory rejected: no human paddle return");
  }
  if (visibleRatio < minVisibleRatio) {
    throw new Error("Trajectory rejected: tab was hidden too much");
  }
  if (header.finalLeftScore > maxPongScore || header.finalRightScore > maxPongScore) {
    throw new Error("Trajectory rejected: impossible final score");
  }
  if (analysis.lastLeftScore !== header.finalLeftScore || analysis.lastRightScore !== header.finalRightScore) {
    throw new Error("Trajectory rejected: header score did not match records");
  }
  if (analysis.visibleSamples !== header.visibleSamples) {
    throw new Error("Trajectory rejected: header visibility count did not match records");
  }
  if (analysis.humanHits !== header.humanHits || analysis.botHits !== header.botHits) {
    throw new Error("Trajectory rejected: header hit counts did not match records");
  }
  if (analysis.inputChanges !== header.inputChanges || analysis.maxRally !== header.maxRally) {
    throw new Error("Trajectory rejected: header summary did not match records");
  }
}

async function createR2Storage() {
  let S3Client;
  let PutObjectCommand;
  try {
    ({ S3Client, PutObjectCommand } = await import("@aws-sdk/client-s3"));
  } catch (error) {
    throw new Error("AIPONG_STORAGE=r2 requires @aws-sdk/client-s3. Run npm install before starting the collector.");
  }

  const accountId = process.env.AIPONG_R2_ACCOUNT_ID || process.env.R2_ACCOUNT_ID || process.env.CLOUDFLARE_ACCOUNT_ID;
  const endpoint = process.env.AIPONG_R2_ENDPOINT || process.env.R2_ENDPOINT || (accountId ? `https://${accountId}.r2.cloudflarestorage.com` : "");
  const bucket = process.env.AIPONG_R2_BUCKET || process.env.R2_BUCKET;
  const accessKeyId = process.env.AIPONG_R2_ACCESS_KEY_ID || process.env.R2_ACCESS_KEY_ID || process.env.AWS_ACCESS_KEY_ID;
  const secretAccessKey = process.env.AIPONG_R2_SECRET_ACCESS_KEY || process.env.R2_SECRET_ACCESS_KEY || process.env.AWS_SECRET_ACCESS_KEY;
  const region = process.env.AIPONG_R2_REGION || process.env.R2_REGION || "auto";
  const prefix = cleanObjectKeyPart(process.env.AIPONG_R2_PREFIX || process.env.R2_PREFIX || "ai-pong-training");

  const missing = [];
  if (!endpoint) missing.push("AIPONG_R2_ACCOUNT_ID or AIPONG_R2_ENDPOINT");
  if (!bucket) missing.push("AIPONG_R2_BUCKET");
  if (!accessKeyId) missing.push("AIPONG_R2_ACCESS_KEY_ID");
  if (!secretAccessKey) missing.push("AIPONG_R2_SECRET_ACCESS_KEY");
  if (missing.length > 0) {
    throw new Error(`Missing R2 configuration: ${missing.join(", ")}`);
  }

  const client = new S3Client({
    endpoint,
    region,
    forcePathStyle: true,
    credentials: {
      accessKeyId,
      secretAccessKey,
    },
  });

  return {
    bucket,
    prefix,
    endpoint,
    async putObject(key, body, contentType) {
      await client.send(new PutObjectCommand({
        Bucket: bucket,
        Key: key,
        Body: body,
        ContentType: contentType,
      }));
    },
  };
}

async function getR2Storage() {
  if (!r2StoragePromise) {
    r2StoragePromise = createR2Storage();
  }
  return r2StoragePromise;
}

function buildTrajectoryNames(header, request, url) {
  const matchId = sanitize(request.headers["x-aipong-match-id"] || url.searchParams.get("match"), "match");
  const chunkIndex = sanitize(request.headers["x-aipong-chunk-index"] || url.searchParams.get("chunk") || header.chunkIndex, "0");
  const reason = sanitize(request.headers["x-aipong-reason"] || url.searchParams.get("reason"), "unknown");
  const date = new Date();
  const day = date.toISOString().slice(0, 10);
  const generation = `checkpoint-${header.checkpointUpdate}`;
  const fileName = `${date.toISOString().replace(/[:.]/g, "-")}-${matchId}-chunk-${chunkIndex}.aipong`;
  const trajectoryFile = joinObjectKey("trajectories", generation, day, fileName);

  return {
    date,
    day,
    generation,
    matchId,
    chunkIndex,
    reason,
    fileName,
    trajectoryFile,
  };
}

async function storeTrajectory(body, header, names) {
  const manifestEntry = {
    createdAt: names.date.toISOString(),
    file: names.trajectoryFile,
    bytes: body.length,
    reason: names.reason,
    matchId: names.matchId,
    chunkIndex: Number(names.chunkIndex),
    storage: storageMode,
    ...header,
  };

  if (storageMode === "local") {
    const fullPath = join(dataRoot, names.trajectoryFile);
    await mkdir(dirname(fullPath), { recursive: true });
    await writeFile(fullPath, body);
    await mkdir(dataRoot, { recursive: true });
    await appendFile(join(dataRoot, "manifest.jsonl"), `${JSON.stringify(manifestEntry)}\n`, "utf8");
    return manifestEntry;
  }

  if (storageMode === "r2") {
    const storage = await getR2Storage();
    const objectKey = joinObjectKey(storage.prefix, names.trajectoryFile);
    const manifestKey = joinObjectKey(storage.prefix, "manifests", names.generation, names.day, names.fileName.replace(/\.aipong$/, ".json"));
    const r2ManifestEntry = {
      ...manifestEntry,
      bucket: storage.bucket,
      file: objectKey,
      manifestFile: manifestKey,
    };

    await storage.putObject(objectKey, body, "application/octet-stream");
    await storage.putObject(manifestKey, `${JSON.stringify(r2ManifestEntry)}\n`, "application/json; charset=utf-8");
    return r2ManifestEntry;
  }

  throw new Error(`Unsupported AIPONG_STORAGE value "${storageMode}". Use "local" or "r2".`);
}

function compareLeaderboardEntries(a, b) {
  return (
    getLeaderboardPoints(b) - getLeaderboardPoints(a) ||
    Number(b.humanWon) - Number(a.humanWon) ||
    b.humanScore - a.humanScore ||
    b.margin - a.margin ||
    b.maxRally - a.maxRally ||
    b.humanHits - a.humanHits ||
    a.createdAt.localeCompare(b.createdAt)
  );
}

function getLeaderboardPoints(entry) {
  const suppliedPoints = Number(entry.points);
  if (Number.isFinite(suppliedPoints)) {
    return Math.max(0, suppliedPoints);
  }

  return (
    clampNumber(entry.humanHits, 0, 100_000) * pointsPerHit +
    clampNumber(entry.maxRally, 0, 100_000) * pointsPerRally +
    clampNumber(entry.humanScore, 0, 99) * pointsPerPoint
  );
}

function getMaxReasonablePoints({ humanScore, humanHits, maxRally }) {
  return (
    clampNumber(humanHits, 0, 100_000) * (pointsPerHit + clampNumber(maxRally, 0, 100_000) * pointsPerRally) +
    clampNumber(humanScore, 0, maxPongScore) * pointsPerPoint
  );
}

async function readLeaderboardEntries() {
  const content = await readFile(leaderboardFile, "utf8").catch((error) => {
    if (error?.code === "ENOENT") {
      return "[]";
    }
    throw error;
  });
  const parsed = JSON.parse(content);
  if (!Array.isArray(parsed)) {
    return [];
  }
  return parsed.sort(compareLeaderboardEntries).slice(0, leaderboardLimit);
}

function normalizeLeaderboardEntry(payload) {
  const humanScore = clampNumber(payload.humanScore, 0, 99);
  const botScore = clampNumber(payload.botScore, 0, 99);
  const humanHits = clampNumber(payload.humanHits, 0, 100_000);
  const botHits = clampNumber(payload.botHits, 0, 100_000);
  const maxRally = clampNumber(payload.maxRally, 0, 100_000);
  const durationMs = clampNumber(payload.durationMs, 0, 24 * 60 * 60 * 1000);
  const fallbackPoints = getLeaderboardPoints({ humanScore, humanHits, maxRally });
  const suppliedPoints = Number(payload.points);
  const points = Number.isFinite(suppliedPoints) ? clampNumber(suppliedPoints, 0, 100_000_000) : fallbackPoints;
  const maxReasonablePoints = getMaxReasonablePoints({ humanScore, humanHits, maxRally });

  if (humanScore > maxPongScore || botScore > maxPongScore) {
    throw new Error("Leaderboard rejected: impossible score");
  }
  if (humanScore < maxPongScore && botScore < maxPongScore) {
    throw new Error("Leaderboard rejected: match was not complete");
  }
  if (durationMs < leaderboardMinDurationMs) {
    throw new Error("Leaderboard rejected: match was too short");
  }
  if (points > maxReasonablePoints + pointsPerPoint) {
    throw new Error("Leaderboard rejected: points were inconsistent with match stats");
  }

  const entry = {
    id: randomUUID(),
    createdAt: new Date().toISOString(),
    name: sanitizePlayerName(payload.name),
    humanScore,
    botScore,
    points,
    margin: humanScore - botScore,
    humanWon: humanScore > botScore,
    humanHits,
    botHits,
    maxRally,
    durationMs,
    modelUpdate: clampNumber(payload.modelUpdate, 0, 10_000_000),
  };
  return entry;
}

async function readJsonBody(request, maxBytes = 20_000) {
  const chunks = [];
  let size = 0;

  return new Promise((resolveBody, rejectBody) => {
    request.on("data", (chunk) => {
      size += chunk.length;
      if (size > maxBytes) {
        request.destroy();
        rejectBody(new Error(`JSON body exceeded ${maxBytes} bytes`));
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => {
      try {
        resolveBody(JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}"));
      } catch {
        rejectBody(new Error("Invalid JSON body"));
      }
    });
    request.on("error", rejectBody);
  });
}

async function handleGetLeaderboard(response) {
  const entries = await readLeaderboardEntries();
  sendJson(response, 200, {
    ok: true,
    entries,
  });
}

async function handlePostLeaderboard(request, response) {
  const payload = await readJsonBody(request);
  const entry = normalizeLeaderboardEntry(payload);

  leaderboardWriteQueue = leaderboardWriteQueue.then(async () => {
    const entries = await readLeaderboardEntries();
    const nextEntries = [...entries, entry].sort(compareLeaderboardEntries).slice(0, leaderboardLimit);
    await mkdir(dataRoot, { recursive: true });
    await writeFile(leaderboardFile, `${JSON.stringify(nextEntries, null, 2)}\n`, "utf8");
    return nextEntries;
  });

  const entries = await leaderboardWriteQueue;
  const usage = await addUsageUnits("leaderboard");
  sendJson(response, 201, {
    ok: true,
    entry,
    entries,
    usage,
  });
}

async function readPlayStats() {
  const content = await readFile(playStatsFile, "utf8").catch((error) => {
    if (error?.code === "ENOENT") {
      return "{}";
    }
    throw error;
  });
  const parsed = JSON.parse(content);
  return {
    plays: clampNumber(parsed.plays, 0, Number.MAX_SAFE_INTEGER),
    updatedAt: parsed.updatedAt || null,
  };
}

async function handleGetStats(response) {
  const stats = await readPlayStats();
  const usage = await getCurrentUsageSnapshot();
  sendJson(response, 200, {
    ok: true,
    ...stats,
    usage,
  });
}

async function handlePostStats(request, response) {
  const payload = await readJsonBody(request).catch(() => ({}));
  const matchId = sanitize(payload.matchId, "");
  const matchKey = matchId ? `${getClientIp(request)}:${matchId}` : "";

  if (!matchId || matchId.length < 8) {
    throw new Error("Play count rejected: missing match id");
  }
  if (recentPlayMatchIds.has(matchKey)) {
    const stats = await readPlayStats();
    const usage = await getCurrentUsageSnapshot();
    sendJson(response, 200, {
      ok: true,
      duplicate: true,
      ...stats,
      usage,
    });
    return;
  }
  recentPlayMatchIds.set(matchKey, Date.now() + 6 * 60 * 60 * 1000);

  playStatsWriteQueue = playStatsWriteQueue.then(async () => {
    const current = await readPlayStats();
    const nextStats = {
      plays: current.plays + 1,
      updatedAt: new Date().toISOString(),
    };
    await mkdir(dataRoot, { recursive: true });
    await writeFile(playStatsFile, `${JSON.stringify(nextStats, null, 2)}\n`, "utf8");
    return nextStats;
  });

  const stats = await playStatsWriteQueue;
  const usage = await addUsageUnits("play");
  sendJson(response, 201, {
    ok: true,
    ...stats,
    usage,
  });
}

async function handleTrajectory(request, response, url) {
  const body = await readBody(request);
  const header = parseTrajectoryHeader(body);
  const names = buildTrajectoryNames(header, request, url);
  const analysis = analyzeTrajectoryRecords(body, header);
  validateTrajectoryQuality(header, analysis, names.reason);
  const manifestEntry = await storeTrajectory(body, header, names);
  const usage = await addUsageUnits("trajectory", body.length);

  sendJson(response, 201, {
    ok: true,
    file: manifestEntry.file,
    sampleCount: header.sampleCount,
    bytes: body.length,
    storage: storageMode,
    usage,
  });
}

async function handleGetRuntime(response) {
  const usage = await getCurrentUsageSnapshot();
  sendJson(response, 200, {
    ok: true,
    usage,
  });
}

async function serveStatic(response, pathname) {
  const usage = await getCurrentUsageSnapshot();
  if (usage.mode === "maintenance" && (pathname === "/" || pathname === "/index.html")) {
    const maintenancePath = resolve(webRoot, "maintenance.html");
    const content = await readFile(maintenancePath, "utf8");
    writeSecurityHeaders(response);
    response.writeHead(503, {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store",
      "Retry-After": String(usageRetryAfterSeconds),
    });
    response.end(content);
    return;
  }

  const requestedPath = pathname === "/" ? "/index.html" : pathname;
  const normalizedPath = normalize(decodeURIComponent(requestedPath)).replace(/^(\.\.[/\\])+/, "");
  const fullPath = resolve(webRoot, `.${normalizedPath}`);

  if (!fullPath.startsWith(webRoot)) {
    writeSecurityHeaders(response);
    response.writeHead(403);
    response.end("Forbidden");
    return;
  }

  const fileStat = await stat(fullPath).catch(() => null);
  if (!fileStat?.isFile()) {
    writeSecurityHeaders(response);
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("Not found");
    return;
  }

  const content = await readFile(fullPath);
  const modelRoot = resolve(webRoot, "models");
  const isModelJson = fullPath.endsWith(".json") && fullPath.startsWith(`${modelRoot}${sep}`);
  const isMutableCurrentModel = fullPath === resolve(modelRoot, "current-policy.json");
  writeSecurityHeaders(response);
  response.writeHead(200, {
    "Content-Type": mimeTypes.get(extname(fullPath)) || "application/octet-stream",
    "Cache-Control": isMutableCurrentModel
      ? "no-cache"
      : isModelJson
      ? "public, max-age=3600"
      : fullPath.endsWith(".json")
        || fullPath.endsWith(".html")
        || fullPath.endsWith(".js")
        || fullPath.endsWith(".css")
        ? "no-cache"
        : "public, max-age=300",
  });
  response.end(content);
}

function shouldSlowRequest(method, pathname) {
  return method === "POST" || pathname === "/" || pathname === "/index.html";
}

if (!["local", "r2"].includes(storageMode)) {
  throw new Error(`Unsupported AIPONG_STORAGE value "${storageMode}". Use "local" or "r2".`);
}

if (process.env.NODE_ENV === "production" && allowedOrigin === "*") {
  console.warn("AIPONG_ALLOWED_ORIGIN is '*' in production. Set it to your deployed origin before public launch.");
}

if (storageMode === "r2") {
  await getR2Storage();
}

setInterval(pruneSmallMaps, Math.max(60_000, rateLimitWindowMs)).unref();

const server = createServer(async (request, response) => {
  const url = new URL(request.url || "/", `http://${request.headers.host || "localhost"}`);

  try {
    if (request.method === "OPTIONS") {
      writeCorsHeaders(response);
      writeSecurityHeaders(response);
      response.writeHead(204);
      response.end();
      return;
    }

    const usage = await getCurrentUsageSnapshot();

    if (request.method === "GET" && url.pathname === "/api/ai-pong/health") {
      sendJson(response, 200, {
        ok: true,
        storageMode,
        usageMode: usage.mode,
        usageEnabled: usage.enabled,
        usagePercentage: usage.percentage,
        ...(exposeHealthDetails
          ? {
              dataRoot,
              maxUploadBytes,
              maxSamplesPerUpload,
              rateLimitWindowMs,
              usage,
            }
          : {}),
      });
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/ai-pong/runtime") {
      await handleGetRuntime(response);
      return;
    }

    if (usage.mode === "maintenance" && request.method === "POST" && url.pathname.startsWith("/api/ai-pong/")) {
      sendMaintenanceJson(response, usage);
      return;
    }

    if (usage.mode === "slow" && shouldSlowRequest(request.method || "GET", url.pathname)) {
      await applyUsageSlowdown(usage);
    }

    if (request.method === "POST" && url.pathname === "/api/ai-pong/trajectory") {
      const rateLimit = checkRateLimit(request, "trajectory", trajectoryRateLimit);
      if (!rateLimit.ok) {
        sendRateLimit(response, rateLimit.retryAfterSeconds);
        return;
      }
      await handleTrajectory(request, response, url);
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/ai-pong/leaderboard") {
      await handleGetLeaderboard(response);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/ai-pong/leaderboard") {
      const rateLimit = checkRateLimit(request, "leaderboard", leaderboardRateLimit);
      if (!rateLimit.ok) {
        sendRateLimit(response, rateLimit.retryAfterSeconds);
        return;
      }
      await handlePostLeaderboard(request, response);
      return;
    }

    if (request.method === "GET" && url.pathname === "/api/ai-pong/stats") {
      await handleGetStats(response);
      return;
    }

    if (request.method === "POST" && url.pathname === "/api/ai-pong/stats") {
      const rateLimit = checkRateLimit(request, "stats", statsRateLimit);
      if (!rateLimit.ok) {
        sendRateLimit(response, rateLimit.retryAfterSeconds);
        return;
      }
      await handlePostStats(request, response);
      return;
    }

    if (request.method === "GET" || request.method === "HEAD") {
      await serveStatic(response, url.pathname);
      return;
    }

    sendJson(response, 405, { ok: false, error: "Method not allowed" });
  } catch (error) {
    sendJson(response, 400, {
      ok: false,
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

server.listen(port, () => {
  console.log(`AI Pong training site: http://localhost:${port}`);
  console.log(`Trajectory storage mode: ${storageMode}`);
  if (storageMode === "local") {
    console.log(`Trajectory data directory: ${dataRoot}`);
  }
});
