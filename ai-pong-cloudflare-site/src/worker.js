const POINTS_PER_HIT = 10;
const POINTS_PER_RALLY = 2;
const POINTS_PER_POINT = 100;
const MAGIC_BYTES = [65, 73, 80, 79, 78, 71, 49, 0];

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    try {
      if (request.method === "OPTIONS") {
        return withHeaders(new Response(null, { status: 204 }), env);
      }

      const usage = await getUsageSnapshot(env);

      if (request.method === "GET" && url.pathname === "/api/ai-pong/health") {
        return jsonResponse({
          ok: true,
          storageMode: "r2+d1",
          usageMode: usage.mode,
          usageEnabled: usage.enabled,
          usagePercentage: usage.percentage,
        }, 200, env);
      }

      if (request.method === "GET" && url.pathname === "/api/ai-pong/runtime") {
        return jsonResponse({ ok: true, usage }, 200, env);
      }

      if (usage.mode === "maintenance" && request.method === "POST" && url.pathname.startsWith("/api/ai-pong/")) {
        return maintenanceJson(usage, env);
      }

      if (usage.mode === "slow" && shouldSlowRequest(request.method, url.pathname)) {
        await sleep(usage.slowdownMs);
      }

      if (request.method === "GET" && url.pathname === "/api/ai-pong/stats") {
        const stats = await readPlayStats(env);
        return jsonResponse({ ok: true, ...stats, usage }, 200, env);
      }

      if (request.method === "POST" && url.pathname === "/api/ai-pong/stats") {
        return handlePostStats(request, env);
      }

      if (request.method === "GET" && url.pathname === "/api/ai-pong/leaderboard") {
        const entries = await readLeaderboardEntries(env);
        return jsonResponse({ ok: true, entries }, 200, env);
      }

      if (request.method === "POST" && url.pathname === "/api/ai-pong/leaderboard") {
        return handlePostLeaderboard(request, env);
      }

      if (request.method === "POST" && url.pathname === "/api/ai-pong/trajectory") {
        return handleTrajectory(request, env, url);
      }

      if (request.method === "GET" || request.method === "HEAD") {
        return serveAsset(request, env, usage);
      }

      return jsonResponse({ ok: false, error: "Method not allowed" }, 405, env);
    } catch (error) {
      return jsonResponse({
        ok: false,
        error: error instanceof Error ? error.message : "Unknown error",
      }, 400, env);
    }
  },
};

function getConfig(env) {
  const slowdownRatio = clampRatio(env.USAGE_SLOWDOWN_RATIO, 0.65);
  const maintenanceRatio = clampRatio(env.USAGE_MAINTENANCE_RATIO, 0.8, slowdownRatio);
  return {
    allowedOrigin: String(env.ALLOWED_ORIGIN || "*"),
    maxUploadBytes: Math.max(0, readNumberEnv(env, "MAX_UPLOAD_BYTES", 150_000)),
    maxSamplesPerUpload: Math.max(1, readNumberEnv(env, "MAX_SAMPLES_PER_UPLOAD", 700)),
    minTrajectorySamples: Math.max(1, readNumberEnv(env, "MIN_TRAJECTORY_SAMPLES", 120)),
    minTrajectoryDurationMs: Math.max(0, readNumberEnv(env, "MIN_TRAJECTORY_DURATION_MS", 20_000)),
    minInputChanges: Math.max(0, readNumberEnv(env, "MIN_INPUT_CHANGES", 4)),
    minHumanHits: Math.max(0, readNumberEnv(env, "MIN_HUMAN_HITS", 1)),
    minVisibleRatio: clampRatio(env.MIN_VISIBLE_RATIO, 0.8),
    maxPongScore: Math.max(1, readNumberEnv(env, "MAX_PONG_SCORE", 7)),
    leaderboardLimit: Math.max(1, readNumberEnv(env, "LEADERBOARD_LIMIT", 25)),
    leaderboardMinDurationMs: Math.max(0, readNumberEnv(env, "LEADERBOARD_MIN_DURATION_MS", 5_000)),
    usageBudgetUnits: Math.max(0, readNumberEnv(env, "USAGE_BUDGET_UNITS", 0)),
    usageWindow: normalizeUsageWindow(env.USAGE_WINDOW),
    usageSlowdownRatio: slowdownRatio,
    usageMaintenanceRatio: maintenanceRatio,
    usageBytesPerUnit: Math.max(1, readNumberEnv(env, "USAGE_BYTES_PER_UNIT", 25_000)),
    usageTrajectoryBaseUnits: Math.max(0, readNumberEnv(env, "USAGE_TRAJECTORY_BASE_UNITS", 1)),
    usagePlayUnits: Math.max(0, readNumberEnv(env, "USAGE_PLAY_UNITS", 1)),
    usageLeaderboardUnits: Math.max(0, readNumberEnv(env, "USAGE_LEADERBOARD_UNITS", 1)),
    usageMinSlowdownMs: Math.max(0, readNumberEnv(env, "USAGE_MIN_SLOWDOWN_MS", 250)),
    usageMaxSlowdownMs: Math.max(0, readNumberEnv(env, "USAGE_MAX_SLOWDOWN_MS", 2_500)),
    usageRetryAfterSeconds: Math.max(60, readNumberEnv(env, "USAGE_RETRY_AFTER_SECONDS", 900)),
  };
}

function readNumberEnv(env, key, fallback) {
  const value = Number(env[key]);
  return Number.isFinite(value) ? value : fallback;
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

function normalizeUsageWindow(value) {
  const normalized = String(value || "monthly").toLowerCase();
  return ["daily", "monthly", "never"].includes(normalized) ? normalized : "monthly";
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

function joinObjectKey(...parts) {
  return parts
    .map((part) =>
      String(part || "")
        .replaceAll("\\", "/")
        .replace(/\.\./g, "")
        .replace(/^\/+|\/+$/g, "")
    )
    .filter(Boolean)
    .join("/");
}

function getClientIp(request) {
  return request.headers.get("CF-Connecting-IP")
    || request.headers.get("x-forwarded-for")?.split(",")[0]?.trim()
    || "unknown";
}

function getUsageWindowKey(config, now = new Date()) {
  const year = now.getUTCFullYear();
  const month = String(now.getUTCMonth() + 1).padStart(2, "0");
  const day = String(now.getUTCDate()).padStart(2, "0");

  if (config.usageWindow === "daily") {
    return `${year}-${month}-${day}`;
  }
  if (config.usageWindow === "never") {
    return "lifetime";
  }
  return `${year}-${month}`;
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

async function readUsageRow(env) {
  const config = getConfig(env);
  const now = new Date();
  const windowKey = getUsageWindowKey(config, now);
  const row = await env.DB.prepare(
    `SELECT window_key, window, updated_at, used_units, trajectory_uploads, trajectory_bytes, leaderboard_posts, play_registrations
     FROM usage_state
     WHERE window_key = ?1`
  ).bind(windowKey).first();

  if (!row || row.window !== config.usageWindow) {
    return {
      window: config.usageWindow,
      windowKey,
      updatedAt: now.toISOString(),
      usedUnits: 0,
      trajectoryUploads: 0,
      trajectoryBytes: 0,
      leaderboardPosts: 0,
      playRegistrations: 0,
    };
  }

  return {
    window: row.window,
    windowKey: row.window_key,
    updatedAt: row.updated_at,
    usedUnits: clampNumber(row.used_units, 0, Number.MAX_SAFE_INTEGER),
    trajectoryUploads: clampNumber(row.trajectory_uploads, 0, Number.MAX_SAFE_INTEGER),
    trajectoryBytes: clampNumber(row.trajectory_bytes, 0, Number.MAX_SAFE_INTEGER),
    leaderboardPosts: clampNumber(row.leaderboard_posts, 0, Number.MAX_SAFE_INTEGER),
    playRegistrations: clampNumber(row.play_registrations, 0, Number.MAX_SAFE_INTEGER),
  };
}

function formatUsageSnapshot(env, row) {
  const config = getConfig(env);

  if (config.usageBudgetUnits <= 0) {
    return {
      enabled: false,
      window: row.window,
      windowKey: row.windowKey,
      budgetUnits: 0,
      usedUnits: row.usedUnits,
      ratio: 0,
      percentage: 0,
      mode: "normal",
      slowdownMs: 0,
      message: describeUsageMode("normal"),
      updatedAt: row.updatedAt,
      trajectoryUploads: row.trajectoryUploads,
      trajectoryBytes: row.trajectoryBytes,
      leaderboardPosts: row.leaderboardPosts,
      playRegistrations: row.playRegistrations,
    };
  }

  const ratio = Math.max(0, row.usedUnits / config.usageBudgetUnits);
  let mode = "normal";
  if (ratio >= config.usageMaintenanceRatio) {
    mode = "maintenance";
  } else if (ratio >= config.usageSlowdownRatio) {
    mode = "slow";
  }

  let slowdownMs = 0;
  if (mode === "slow") {
    const progress = config.usageMaintenanceRatio > config.usageSlowdownRatio
      ? (ratio - config.usageSlowdownRatio) / (config.usageMaintenanceRatio - config.usageSlowdownRatio)
      : 1;
    slowdownMs = Math.round(
      config.usageMinSlowdownMs + clampNumber(progress, 0, 1) * (config.usageMaxSlowdownMs - config.usageMinSlowdownMs)
    );
  }

  return {
    enabled: true,
    window: row.window,
    windowKey: row.windowKey,
    budgetUnits: config.usageBudgetUnits,
    usedUnits: row.usedUnits,
    ratio,
    percentage: Math.round(ratio * 1000) / 10,
    mode,
    slowdownMs,
    message: describeUsageMode(mode),
    updatedAt: row.updatedAt,
    trajectoryUploads: row.trajectoryUploads,
    trajectoryBytes: row.trajectoryBytes,
    leaderboardPosts: row.leaderboardPosts,
    playRegistrations: row.playRegistrations,
  };
}

async function getUsageSnapshot(env) {
  return formatUsageSnapshot(env, await readUsageRow(env));
}

async function addUsageUnits(env, kind, bytes = 0) {
  const config = getConfig(env);
  const current = await getUsageSnapshot(env);
  if (!current.enabled || current.mode === "maintenance") {
    return current;
  }

  let unitsAdded = 0;
  let trajectoryUploads = 0;
  let trajectoryBytes = 0;
  let leaderboardPosts = 0;
  let playRegistrations = 0;

  if (kind === "trajectory") {
    unitsAdded = config.usageTrajectoryBaseUnits + Math.ceil(Math.max(0, bytes) / config.usageBytesPerUnit);
    trajectoryUploads = 1;
    trajectoryBytes = Math.max(0, bytes);
  } else if (kind === "leaderboard") {
    unitsAdded = config.usageLeaderboardUnits;
    leaderboardPosts = 1;
  } else if (kind === "play") {
    unitsAdded = config.usagePlayUnits;
    playRegistrations = 1;
  }

  const now = new Date();
  const windowKey = getUsageWindowKey(config, now);
  await env.DB.prepare(
    `INSERT INTO usage_state (
       window_key, window, updated_at, used_units, trajectory_uploads, trajectory_bytes, leaderboard_posts, play_registrations
     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
     ON CONFLICT(window_key) DO UPDATE SET
       updated_at = excluded.updated_at,
       used_units = usage_state.used_units + excluded.used_units,
       trajectory_uploads = usage_state.trajectory_uploads + excluded.trajectory_uploads,
       trajectory_bytes = usage_state.trajectory_bytes + excluded.trajectory_bytes,
       leaderboard_posts = usage_state.leaderboard_posts + excluded.leaderboard_posts,
       play_registrations = usage_state.play_registrations + excluded.play_registrations`
  ).bind(
    windowKey,
    config.usageWindow,
    now.toISOString(),
    unitsAdded,
    trajectoryUploads,
    trajectoryBytes,
    leaderboardPosts,
    playRegistrations
  ).run();

  return getUsageSnapshot(env);
}

function shouldSlowRequest(method, pathname) {
  return method === "POST" || pathname === "/" || pathname === "/index.html";
}

function sleep(ms) {
  if (!ms || ms <= 0) {
    return Promise.resolve();
  }
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function buildHeaders(env, extraHeaders = {}) {
  const config = getConfig(env);
  const headers = new Headers({
    "Access-Control-Allow-Origin": config.allowedOrigin,
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,X-Aipong-Match-Id,X-Aipong-Chunk-Index,X-Aipong-Reason",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Content-Security-Policy": "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
  });

  for (const [key, value] of Object.entries(extraHeaders)) {
    headers.set(key, value);
  }

  return headers;
}

function withHeaders(response, env, extraHeaders = {}) {
  const headers = buildHeaders(env, extraHeaders);
  for (const [key, value] of response.headers.entries()) {
    headers.set(key, value);
  }

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function jsonResponse(payload, status, env, extraHeaders = {}) {
  return withHeaders(new Response(JSON.stringify(payload), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      ...extraHeaders,
    },
  }), env);
}

function maintenanceJson(usage, env) {
  const config = getConfig(env);
  return jsonResponse({
    ok: false,
    mode: "maintenance",
    error: usage.message,
    usage,
  }, 503, env, {
    "Retry-After": String(config.usageRetryAfterSeconds),
  });
}

function getCacheControlForPath(pathname) {
  if (pathname === "/models/current-policy.json") {
    return "no-cache";
  }
  if (pathname.endsWith(".json") && pathname.startsWith("/models/")) {
    return "public, max-age=3600";
  }
  if (pathname.endsWith(".html") || pathname.endsWith(".js") || pathname.endsWith(".css") || pathname.endsWith(".json")) {
    return "no-cache";
  }
  return "public, max-age=300";
}

async function serveAsset(request, env, usage) {
  const url = new URL(request.url);
  if (usage.mode === "maintenance" && (url.pathname === "/" || url.pathname === "/index.html")) {
    const maintenanceRequest = new Request(new URL("/maintenance.html", request.url), request);
    const assetResponse = await env.ASSETS.fetch(maintenanceRequest);
    return withHeaders(new Response(assetResponse.body, {
      status: 503,
      headers: assetResponse.headers,
    }), env, {
      "Cache-Control": "no-store",
      "Retry-After": String(getConfig(env).usageRetryAfterSeconds),
    });
  }

  const assetResponse = await env.ASSETS.fetch(request);
  return withHeaders(assetResponse, env, {
    "Cache-Control": getCacheControlForPath(url.pathname === "/" ? "/index.html" : url.pathname),
  });
}

async function readPlayStats(env) {
  const row = await env.DB.prepare(
    "SELECT plays, updated_at FROM play_totals WHERE key = 'global'"
  ).first();

  return {
    plays: clampNumber(row?.plays, 0, Number.MAX_SAFE_INTEGER),
    updatedAt: row?.updated_at || null,
  };
}

async function handlePostStats(request, env) {
  const payload = await request.json().catch(() => ({}));
  const matchId = sanitize(payload.matchId, "");
  if (!matchId || matchId.length < 8) {
    throw new Error("Play count rejected: missing match id");
  }

  const matchKey = `${getClientIp(request)}:${matchId}`;
  const now = new Date().toISOString();
  const insertResult = await env.DB.prepare(
    `INSERT INTO played_matches (match_key, created_at)
     VALUES (?1, ?2)
     ON CONFLICT(match_key) DO NOTHING`
  ).bind(matchKey, now).run();

  if (insertResult.meta?.changes === 0) {
    const stats = await readPlayStats(env);
    const usage = await getUsageSnapshot(env);
    return jsonResponse({ ok: true, duplicate: true, ...stats, usage }, 200, env);
  }

  await env.DB.prepare(
    `INSERT INTO play_totals (key, plays, updated_at)
     VALUES ('global', 1, ?1)
     ON CONFLICT(key) DO UPDATE SET
       plays = play_totals.plays + 1,
       updated_at = excluded.updated_at`
  ).bind(now).run();

  const stats = await readPlayStats(env);
  const usage = await addUsageUnits(env, "play");
  return jsonResponse({ ok: true, ...stats, usage }, 201, env);
}

function getLeaderboardPoints(entry) {
  const suppliedPoints = Number(entry.points);
  if (Number.isFinite(suppliedPoints)) {
    return Math.max(0, suppliedPoints);
  }
  return (
    clampNumber(entry.humanHits, 0, 100_000) * POINTS_PER_HIT +
    clampNumber(entry.maxRally, 0, 100_000) * POINTS_PER_RALLY +
    clampNumber(entry.humanScore, 0, 99) * POINTS_PER_POINT
  );
}

function getMaxReasonablePoints(entry, config) {
  return (
    clampNumber(entry.humanHits, 0, 100_000) * (POINTS_PER_HIT + clampNumber(entry.maxRally, 0, 100_000) * POINTS_PER_RALLY) +
    clampNumber(entry.humanScore, 0, config.maxPongScore) * POINTS_PER_POINT
  );
}

async function readLeaderboardEntries(env) {
  const config = getConfig(env);
  const result = await env.DB.prepare(
    `SELECT id, created_at, name, human_score, bot_score, points, margin, human_won, human_hits, bot_hits, max_rally, duration_ms, model_update
     FROM leaderboard_entries
     ORDER BY points DESC, human_won DESC, human_score DESC, margin DESC, max_rally DESC, human_hits DESC, created_at ASC
     LIMIT ?1`
  ).bind(config.leaderboardLimit).all();

  return (result.results || []).map((row) => ({
    id: row.id,
    createdAt: row.created_at,
    name: row.name,
    humanScore: row.human_score,
    botScore: row.bot_score,
    points: row.points,
    margin: row.margin,
    humanWon: Boolean(row.human_won),
    humanHits: row.human_hits,
    botHits: row.bot_hits,
    maxRally: row.max_rally,
    durationMs: row.duration_ms,
    modelUpdate: row.model_update,
  }));
}

function normalizeLeaderboardEntry(payload, env) {
  const config = getConfig(env);
  const humanScore = clampNumber(payload.humanScore, 0, 99);
  const botScore = clampNumber(payload.botScore, 0, 99);
  const humanHits = clampNumber(payload.humanHits, 0, 100_000);
  const botHits = clampNumber(payload.botHits, 0, 100_000);
  const maxRally = clampNumber(payload.maxRally, 0, 100_000);
  const durationMs = clampNumber(payload.durationMs, 0, 24 * 60 * 60 * 1000);
  const fallbackPoints = getLeaderboardPoints({ humanScore, humanHits, maxRally });
  const suppliedPoints = Number(payload.points);
  const points = Number.isFinite(suppliedPoints) ? clampNumber(suppliedPoints, 0, 100_000_000) : fallbackPoints;
  const maxReasonablePoints = getMaxReasonablePoints({ humanScore, humanHits, maxRally }, config);

  if (humanScore > config.maxPongScore || botScore > config.maxPongScore) {
    throw new Error("Leaderboard rejected: impossible score");
  }
  if (humanScore < config.maxPongScore && botScore < config.maxPongScore) {
    throw new Error("Leaderboard rejected: match was not complete");
  }
  if (durationMs < config.leaderboardMinDurationMs) {
    throw new Error("Leaderboard rejected: match was too short");
  }
  if (points > maxReasonablePoints + POINTS_PER_POINT) {
    throw new Error("Leaderboard rejected: points were inconsistent with match stats");
  }

  return {
    id: crypto.randomUUID(),
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
}

async function handlePostLeaderboard(request, env) {
  const payload = await request.json();
  const entry = normalizeLeaderboardEntry(payload, env);
  await env.DB.prepare(
    `INSERT INTO leaderboard_entries (
       id, created_at, name, human_score, bot_score, points, margin, human_won, human_hits, bot_hits, max_rally, duration_ms, model_update
     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)`
  ).bind(
    entry.id,
    entry.createdAt,
    entry.name,
    entry.humanScore,
    entry.botScore,
    entry.points,
    entry.margin,
    entry.humanWon ? 1 : 0,
    entry.humanHits,
    entry.botHits,
    entry.maxRally,
    entry.durationMs,
    entry.modelUpdate
  ).run();

  const entries = await readLeaderboardEntries(env);
  const usage = await addUsageUnits(env, "leaderboard");
  return jsonResponse({ ok: true, entry, entries, usage }, 201, env);
}

function parseTrajectoryHeader(bytes, env) {
  if (bytes.byteLength < 64) {
    throw new Error("Body is smaller than the 64-byte trajectory header");
  }

  for (let index = 0; index < MAGIC_BYTES.length; index += 1) {
    if (bytes[index] !== MAGIC_BYTES[index]) {
      throw new Error("Invalid AIPONG magic header");
    }
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const config = getConfig(env);
  const header = {
    version: view.getUint16(8, true),
    headerBytes: view.getUint16(10, true),
    recordBytes: view.getUint16(12, true),
    sampleHz: view.getUint16(14, true),
    sampleCount: view.getUint32(16, true),
    checkpointUpdate: view.getUint32(20, true),
    durationMs: view.getUint32(24, true),
    finalLeftScore: view.getUint16(28, true),
    finalRightScore: view.getUint16(30, true),
    humanHits: view.getUint16(32, true),
    botHits: view.getUint16(34, true),
    maxRally: view.getUint16(36, true),
    inputChanges: view.getUint16(38, true),
    visibleSamples: view.getUint16(40, true),
    flags: view.getUint16(42, true),
    chunkIndex: view.getUint32(44, true),
  };

  if (header.version !== 1) {
    throw new Error(`Unsupported trajectory version ${header.version}`);
  }
  if (header.headerBytes !== 64) {
    throw new Error(`Unexpected header size ${header.headerBytes}`);
  }
  if (header.recordBytes !== 48) {
    throw new Error(`Unexpected record size ${header.recordBytes}`);
  }
  if (header.sampleHz !== 10) {
    throw new Error(`Unexpected sample rate ${header.sampleHz}`);
  }
  if (header.sampleCount < 1 || header.sampleCount > config.maxSamplesPerUpload) {
    throw new Error(`Unreasonable sample count ${header.sampleCount}`);
  }

  const expectedBytes = header.headerBytes + header.sampleCount * header.recordBytes;
  if (bytes.byteLength !== expectedBytes) {
    throw new Error(`Expected ${expectedBytes} bytes, received ${bytes.byteLength}`);
  }

  return header;
}

function analyzeTrajectoryRecords(bytes, header, env) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const config = getConfig(env);
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
    const humanAction = view.getUint8(offset + 36);
    const botAction = view.getUint8(offset + 37);
    const flags = view.getUint8(offset + 42);
    const leftScore = view.getUint8(offset + 43);
    const rightScore = view.getUint8(offset + 44);
    const rallyHits = view.getUint16(offset + 45, true);
    const visible = view.getUint8(offset + 47);

    if (humanAction > 2 || botAction > 2) {
      throw new Error("Trajectory contains invalid paddle action values");
    }
    if (leftScore > config.maxPongScore || rightScore > config.maxPongScore) {
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

function validateTrajectoryQuality(header, analysis, reason, env) {
  const config = getConfig(env);
  const visibleRatio = header.sampleCount > 0 ? analysis.visibleSamples / header.sampleCount : 0;

  if (header.sampleCount < config.minTrajectorySamples) {
    throw new Error(`Trajectory rejected: too few samples (${header.sampleCount})`);
  }
  if (reason !== "minute" && header.durationMs < config.minTrajectoryDurationMs) {
    throw new Error("Trajectory rejected: match segment was too short");
  }
  if (analysis.inputChanges < config.minInputChanges) {
    throw new Error("Trajectory rejected: too little human input variation");
  }
  if (analysis.humanHits < config.minHumanHits) {
    throw new Error("Trajectory rejected: no human paddle return");
  }
  if (visibleRatio < config.minVisibleRatio) {
    throw new Error("Trajectory rejected: tab was hidden too much");
  }
  if (header.finalLeftScore > config.maxPongScore || header.finalRightScore > config.maxPongScore) {
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

function buildTrajectoryNames(header, request, url) {
  const matchId = sanitize(request.headers.get("X-Aipong-Match-Id") || url.searchParams.get("match"), "match");
  const chunkIndex = sanitize(request.headers.get("X-Aipong-Chunk-Index") || url.searchParams.get("chunk") || header.chunkIndex, "0");
  const reason = sanitize(request.headers.get("X-Aipong-Reason") || url.searchParams.get("reason"), "unknown");
  const date = new Date();
  const day = date.toISOString().slice(0, 10);
  const generation = `checkpoint-${header.checkpointUpdate}`;
  const fileName = `${date.toISOString().replace(/[:.]/g, "-")}-${matchId}-chunk-${chunkIndex}.aipong`;
  const objectKey = joinObjectKey("trajectories", generation, day, fileName);

  return {
    date,
    generation,
    matchId,
    chunkIndex,
    reason,
    fileName,
    objectKey,
  };
}

async function handleTrajectory(request, env, url) {
  const config = getConfig(env);
  const arrayBuffer = await request.arrayBuffer();
  if (arrayBuffer.byteLength > config.maxUploadBytes) {
    throw new Error(`Upload exceeded ${config.maxUploadBytes} bytes`);
  }

  const bytes = new Uint8Array(arrayBuffer);
  const header = parseTrajectoryHeader(bytes, env);
  const names = buildTrajectoryNames(header, request, url);
  const analysis = analyzeTrajectoryRecords(bytes, header, env);
  validateTrajectoryQuality(header, analysis, names.reason, env);

  await env.TRAINING_BUCKET.put(names.objectKey, arrayBuffer, {
    httpMetadata: {
      contentType: "application/octet-stream",
    },
  });

  await env.DB.prepare(
    `INSERT INTO trajectory_manifest (
       id, created_at, object_key, bytes, reason, match_id, chunk_index, checkpoint_update, sample_count, duration_ms
     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)`
  ).bind(
    crypto.randomUUID(),
    names.date.toISOString(),
    names.objectKey,
    arrayBuffer.byteLength,
    names.reason,
    names.matchId,
    Number(names.chunkIndex),
    header.checkpointUpdate,
    header.sampleCount,
    header.durationMs
  ).run();

  const usage = await addUsageUnits(env, "trajectory", arrayBuffer.byteLength);
  return jsonResponse({
    ok: true,
    file: names.objectKey,
    sampleCount: header.sampleCount,
    bytes: arrayBuffer.byteLength,
    storage: "r2",
    usage,
  }, 201, env);
}
