const MODEL_URL = "/models/current-policy.json";
const LOG_ENDPOINT = "/api/ai-pong/trajectory";
const LEADERBOARD_ENDPOINT = "/api/ai-pong/leaderboard";
const STATS_ENDPOINT = "/api/ai-pong/stats";
const PLAYER_NAME_STORAGE_KEY = "ai-pong-player-name";
const STATS_REFRESH_MS = 60_000;

const SAMPLE_HZ = 10;
const SAMPLE_INTERVAL = 1 / SAMPLE_HZ;
const MAX_CHUNK_SECONDS = 60;
const MAX_SAMPLES_PER_CHUNK = SAMPLE_HZ * MAX_CHUNK_SECONDS;
const OBS_SCALE = 32767;
const REWARD_SCALE = 1000;
const HEADER_BYTES = 64;
const RECORD_BYTES = 48;
const FORMAT_VERSION = 1;
const MAGIC = "AIPONG1\0";

const MIN_GOOD_DURATION_MS = 20_000;
const MIN_GOOD_SAMPLES = 120;
const MIN_INPUT_CHANGES = 4;
const MIN_HUMAN_HITS = 1;
const MIN_VISIBLE_RATIO = 0.8;
const POINTS_PER_HIT = 10;
const POINTS_PER_RALLY = 2;
const POINTS_PER_POINT = 100;

const fallbackConfig = {
  width: 900,
  height: 600,
  paddle_width: 16,
  paddle_height: 96,
  paddle_margin: 36,
  paddle_speed: 540,
  ball_radius: 9,
  serve_speed: 360,
  max_ball_speed: 920,
  ball_speedup: 1.045,
  max_bounce_angle: (65 * Math.PI) / 180,
  dt: 1 / 60,
  score_to_win: 7,
  hit_reward: 0.05,
};

const canvas = document.querySelector("#pong-canvas");
const statusText = document.querySelector("#status");
const scoreText = document.querySelector("#score");
const trainingStatusText = document.querySelector("#training-status");
const runtimeBanner = document.querySelector("#runtime-banner");
const sampleMeterFill = document.querySelector("#sample-meter-fill");
const sampleCountText = document.querySelector("#sample-count");
const playerForm = document.querySelector("#player-form");
const playerNameInput = document.querySelector("#player-name");
const playerStatusText = document.querySelector("#player-status");
const playerSaveButton = playerForm.querySelector("button[type='submit']");
const leaderboardList = document.querySelector("#leaderboard-list");
const leaderboardStatusText = document.querySelector("#leaderboard-status");
const leaderboardRefreshButton = document.querySelector("#leaderboard-refresh");
const playCountText = document.querySelector("#play-count");
const resetButton = document.querySelector("#reset-button");
const context = canvas.getContext("2d");

let policy = null;
let config = fallbackConfig;
let game = createInitialState(config);
let lastTime = performance.now();
let accumulator = 0;
let sampleAccumulator = 0;
let overlayMessage = "Loading bot...";
let overlayUntil = performance.now() + 2400;
let pointerActive = false;
let pointerTarget = null;
let keyState = { up: false, down: false };
let lastHudUpdate = 0;
let matchId = createMatchId();
let chunkIndex = 0;
let chunk = createChunk();
let transientTrainingMessage = "";
let transientTrainingUntil = 0;
let matchStartedAt = performance.now();
let playerName = "";
let playRegisteredForMatch = false;
let runtimeState = {
  enabled: false,
  mode: "normal",
  percentage: 0,
  slowdownMs: 0,
  message: "",
};

function compileLayer(layer) {
  return {
    weight: layer.weight.map((row) => Float32Array.from(row)),
    bias: Float32Array.from(layer.bias),
  };
}

function compilePolicy(model) {
  return {
    meta: model,
    l0: compileLayer(model.layers.backbone0),
    l1: compileLayer(model.layers.backbone2),
    policy: compileLayer(model.layers.policyHead),
  };
}

function dense(layer, input, activate = false) {
  const output = new Float32Array(layer.bias.length);

  for (let rowIndex = 0; rowIndex < layer.weight.length; rowIndex += 1) {
    const row = layer.weight[rowIndex];
    let sum = layer.bias[rowIndex];

    for (let columnIndex = 0; columnIndex < row.length; columnIndex += 1) {
      sum += row[columnIndex] * input[columnIndex];
    }

    output[rowIndex] = activate ? Math.tanh(sum) : sum;
  }

  return output;
}

function choosePolicyAction(currentPolicy, observation) {
  const hidden0 = dense(currentPolicy.l0, observation, true);
  const hidden1 = dense(currentPolicy.l1, hidden0, true);
  const logits = dense(currentPolicy.policy, hidden1);
  let bestIndex = 0;
  let bestLogit = logits[0];

  for (let index = 1; index < logits.length; index += 1) {
    if (logits[index] > bestLogit) {
      bestIndex = index;
      bestLogit = logits[index];
    }
  }

  return bestIndex;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function createInitialState(envConfig) {
  const state = {
    leftY: envConfig.height / 2,
    rightY: envConfig.height / 2,
    ballX: envConfig.width / 2,
    ballY: envConfig.height / 2,
    ballVx: 0,
    ballVy: 0,
    leftScore: 0,
    rightScore: 0,
    leftHits: 0,
    rightHits: 0,
    rallyHits: 0,
    maxRally: 0,
    points: 0,
  };

  serveBall(state, envConfig);
  return state;
}

function serveBall(state, envConfig, serveDirection) {
  const angle = (Math.random() * 2 - 1) * 0.32;
  const direction = serveDirection ?? (Math.random() < 0.5 ? -1 : 1);

  state.ballX = envConfig.width / 2;
  state.ballY = envConfig.height / 2 + (Math.random() * 2 - 1) * (envConfig.height * 0.15);
  state.ballVx = direction * envConfig.serve_speed * Math.cos(angle);
  state.ballVy = envConfig.serve_speed * Math.sin(angle);
  state.rallyHits = 0;
}

function movePaddle(paddleY, action, envConfig) {
  const directionByAction = [0, -1, 1];
  const halfPaddle = envConfig.paddle_height / 2;

  return clamp(
    paddleY + directionByAction[action] * envConfig.paddle_speed * envConfig.dt,
    halfPaddle,
    envConfig.height - halfPaddle
  );
}

function reflectFromPaddle(state, paddleY, direction, envConfig) {
  const offset = clamp((state.ballY - paddleY) / (envConfig.paddle_height / 2), -1, 1);
  const speed = Math.min(
    Math.hypot(state.ballVx, state.ballVy) * envConfig.ball_speedup,
    envConfig.max_ball_speed
  );
  const angle = offset * envConfig.max_bounce_angle;

  state.ballVx = direction * speed * Math.cos(angle);
  state.ballVy = speed * Math.sin(angle);
}

function getLeftObservation(state, envConfig) {
  return Float32Array.from([
    state.leftY / envConfig.height,
    state.rightY / envConfig.height,
    state.ballX / envConfig.width,
    state.ballY / envConfig.height,
    state.ballVx / envConfig.max_ball_speed,
    state.ballVy / envConfig.max_ball_speed,
    (state.ballY - state.leftY) / envConfig.height,
    (state.ballY - state.rightY) / envConfig.height,
    (state.leftScore - state.rightScore) / envConfig.score_to_win,
  ]);
}

function getRightObservation(state, envConfig) {
  return Float32Array.from([
    state.rightY / envConfig.height,
    state.leftY / envConfig.height,
    (envConfig.width - state.ballX) / envConfig.width,
    state.ballY / envConfig.height,
    -state.ballVx / envConfig.max_ball_speed,
    state.ballVy / envConfig.max_ball_speed,
    (state.ballY - state.rightY) / envConfig.height,
    (state.ballY - state.leftY) / envConfig.height,
    (state.rightScore - state.leftScore) / envConfig.score_to_win,
  ]);
}

function chooseTrackerAction(ballY, paddleY) {
  if (Math.abs(ballY - paddleY) < 8) {
    return 0;
  }

  return ballY < paddleY ? 1 : 2;
}

function currentHumanAction() {
  if (keyState.up !== keyState.down) {
    return keyState.up ? 1 : 2;
  }

  if (pointerActive && pointerTarget !== null) {
    if (Math.abs(pointerTarget - game.leftY) < 8) {
      return 0;
    }

    return pointerTarget < game.leftY ? 1 : 2;
  }

  return 0;
}

function createMatchId() {
  const random = new Uint32Array(2);
  crypto.getRandomValues(random);
  return `${Date.now().toString(36)}-${random[0].toString(36)}${random[1].toString(36)}`;
}

function sanitizePlayerName(value) {
  return String(value || "")
    .trim()
    .replace(/\s+/g, " ")
    .replace(/[^\w .-]/g, "")
    .slice(0, 20);
}

function setPlayerName(name) {
  playerName = sanitizePlayerName(name);
  playerNameInput.value = playerName;

  if (playerName) {
    localStorage.setItem(PLAYER_NAME_STORAGE_KEY, playerName);
    playerStatusText.textContent = `Playing as ${playerName}`;
    return;
  }

  localStorage.removeItem(PLAYER_NAME_STORAGE_KEY);
  playerStatusText.textContent = "Set a name to claim the leaderboard.";
}

function initializePlayerName() {
  const storedName = localStorage.getItem(PLAYER_NAME_STORAGE_KEY) || "";
  setPlayerName(storedName);
  if (!storedName) {
    requestAnimationFrame(() => playerNameInput.focus({ preventScroll: true }));
  }
}

function renderLeaderboard(entries) {
  leaderboardList.innerHTML = "";

  if (!entries.length) {
    const emptyItem = document.createElement("li");
    emptyItem.textContent = "No completed matches yet.";
    leaderboardList.append(emptyItem);
    leaderboardStatusText.textContent = "Finish a match to claim the board.";
    return;
  }

  entries.slice(0, 8).forEach((entry) => {
    const item = document.createElement("li");
    const name = document.createElement("span");
    const score = document.createElement("span");
    const detail = document.createElement("small");

    name.textContent = entry.name;
    score.textContent = formatPoints(entry.points ?? estimateLegacyPoints(entry));
    detail.textContent = `pong ${entry.humanScore}-${entry.botScore} · rally ${entry.maxRally} · hits ${entry.humanHits}`;

    item.append(name, score, detail);
    leaderboardList.append(item);
  });

  leaderboardStatusText.textContent = `Top ${Math.min(entries.length, 8)} point scores`;
}

function formatPoints(points) {
  return `${new Intl.NumberFormat().format(Math.max(0, Math.round(Number(points) || 0)))} pts`;
}

function estimateLegacyPoints(entry) {
  return (Number(entry.humanHits) || 0) * POINTS_PER_HIT
    + (Number(entry.maxRally) || 0) * POINTS_PER_RALLY
    + (Number(entry.humanScore) || 0) * POINTS_PER_POINT;
}

function formatPlayCount(plays) {
  const count = Number.isFinite(plays) ? plays : 0;
  return `${new Intl.NumberFormat().format(count)} ${count === 1 ? "game" : "games"}`;
}

function renderPlayCount(payload) {
  if (!playCountText) {
    return;
  }
  playCountText.textContent = formatPlayCount(Number(payload?.plays || 0));
}

function readJsonResponse(response) {
  return response.json().catch(() => ({}));
}

function syncRuntimeControls() {
  const isMaintenance = runtimeState.mode === "maintenance";
  resetButton.disabled = isMaintenance;
  leaderboardRefreshButton.disabled = isMaintenance;
  playerSaveButton.disabled = isMaintenance;
}

function renderRuntimeState() {
  const mode = runtimeState.mode || "normal";
  const isSlow = mode === "slow";
  const isMaintenance = mode === "maintenance";

  if (runtimeBanner) {
    runtimeBanner.hidden = !isSlow && !isMaintenance;
    runtimeBanner.dataset.state = mode;
    runtimeBanner.textContent = runtimeState.message || "";
  }

  syncRuntimeControls();
}

function applyRuntimePayload(payload) {
  const usage = payload?.usage || payload;
  if (!usage || typeof usage !== "object") {
    return;
  }

  runtimeState = {
    enabled: Boolean(usage.enabled),
    mode: usage.mode || "normal",
    percentage: Number(usage.percentage || 0),
    slowdownMs: Number(usage.slowdownMs || 0),
    message: String(usage.message || ""),
  };

  renderRuntimeState();
}

function loadPlayStats() {
  fetch(STATS_ENDPOINT)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`stats returned ${response.status}`);
      }
      return readJsonResponse(response);
    })
    .then((payload) => {
      renderPlayCount(payload);
      applyRuntimePayload(payload);
    })
    .catch((error) => {
      console.error(error);
      if (playCountText) {
        playCountText.textContent = "Live count unavailable";
      }
    });
}

function registerPlay() {
  if (playRegisteredForMatch || runtimeState.mode === "maintenance") {
    return;
  }
  playRegisteredForMatch = true;

  fetch(STATS_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      matchId,
      modelUpdate: policy?.meta.checkpointUpdate ?? 0,
    }),
  })
    .then((response) => {
      return readJsonResponse(response).then((payload) => ({ response, payload }));
    })
    .then(({ response, payload }) => {
      applyRuntimePayload(payload);
      if (!response.ok) {
        throw new Error(`stats returned ${response.status}`);
      }
      return payload;
    })
    .then(renderPlayCount)
    .catch((error) => {
      console.error(error);
      playRegisteredForMatch = false;
    });
}

function loadLeaderboard() {
  fetch(LEADERBOARD_ENDPOINT)
    .then((response) => {
      return readJsonResponse(response).then((payload) => ({ response, payload }));
    })
    .then(({ response, payload }) => {
      if (!response.ok) {
        throw new Error(`leaderboard returned ${response.status}`);
      }
      renderLeaderboard(payload.entries || []);
    })
    .catch((error) => {
      console.error(error);
      leaderboardList.innerHTML = "<li>Leaderboard unavailable.</li>";
      leaderboardStatusText.textContent = "Leaderboard unavailable.";
    });
}

function submitLeaderboardEntry(summary) {
  if (!playerName || runtimeState.mode === "maintenance") {
    playerStatusText.textContent = "Set a name to claim the leaderboard.";
    return;
  }

  fetch(LEADERBOARD_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      name: playerName,
      ...summary,
    }),
  })
    .then((response) => {
      return readJsonResponse(response).then((payload) => ({ response, payload }));
    })
    .then(({ response, payload }) => {
      applyRuntimePayload(payload);
      if (!response.ok) {
        throw new Error(`leaderboard returned ${response.status}`);
      }
      renderLeaderboard(payload.entries || []);
      leaderboardStatusText.textContent = `Score saved for ${playerName}`;
    })
    .catch((error) => {
      console.error(error);
      leaderboardStatusText.textContent = "Score could not be saved.";
    });
}

function createChunk() {
  return {
    samples: [],
    startedAt: performance.now(),
    visibleSamples: 0,
    inputChanges: 0,
    lastHumanAction: 0,
    humanHits: 0,
    botHits: 0,
    maxRally: 0,
  };
}

function setTrainingMessage(message, ttlMs = 2600) {
  transientTrainingMessage = message;
  transientTrainingUntil = performance.now() + ttlMs;
  if (trainingStatusText) {
    trainingStatusText.textContent = message;
  }
}

function quantizeUnit(value) {
  return clamp(Math.round(clamp(value, -1, 1) * OBS_SCALE), -32768, 32767);
}

function quantizeReward(value) {
  return clamp(Math.round(value * REWARD_SCALE), -32768, 32767);
}

function createRecord(humanObservation, botObservation, humanAction, botAction, stepResult) {
  return {
    humanObservation: Array.from(humanObservation),
    botObservation: Array.from(botObservation),
    humanAction,
    botAction,
    humanReward: stepResult.humanReward,
    botReward: stepResult.botReward,
    flags: stepResult.flags,
    leftScore: game.leftScore,
    rightScore: game.rightScore,
    rallyHits: game.rallyHits,
    visible: document.visibilityState === "visible",
  };
}

function addSample(record) {
  if (chunk.samples.length === 0) {
    chunk.lastHumanAction = record.humanAction;
  } else if (record.humanAction !== chunk.lastHumanAction) {
    chunk.inputChanges += 1;
    chunk.lastHumanAction = record.humanAction;
  }

  if (record.flags & 2) {
    chunk.humanHits += 1;
  }
  if (record.flags & 4) {
    chunk.botHits += 1;
  }
  if (record.visible) {
    chunk.visibleSamples += 1;
  }

  chunk.maxRally = Math.max(chunk.maxRally, record.rallyHits);
  chunk.samples.push(record);

  if (chunk.samples.length >= MAX_SAMPLES_PER_CHUNK) {
    flushChunk("minute");
  }
}

function shouldSendChunk(reason) {
  const durationMs = performance.now() - chunk.startedAt;
  const visibleRatio = chunk.samples.length > 0 ? chunk.visibleSamples / chunk.samples.length : 0;

  if (!policy) {
    return { ok: false, why: "bot not loaded" };
  }
  if (chunk.samples.length < MIN_GOOD_SAMPLES) {
    return { ok: false, why: `too few samples (${chunk.samples.length})` };
  }
  if (durationMs < MIN_GOOD_DURATION_MS && reason !== "minute") {
    return { ok: false, why: "too short" };
  }
  if (chunk.inputChanges < MIN_INPUT_CHANGES) {
    return { ok: false, why: "too little input variation" };
  }
  if (chunk.humanHits < MIN_HUMAN_HITS) {
    return { ok: false, why: "no human paddle return" };
  }
  if (visibleRatio < MIN_VISIBLE_RATIO) {
    return { ok: false, why: "tab was hidden too much" };
  }

  return { ok: true, why: "ok" };
}

function encodeChunk(reason) {
  const sampleCount = chunk.samples.length;
  const buffer = new ArrayBuffer(HEADER_BYTES + sampleCount * RECORD_BYTES);
  const view = new DataView(buffer);
  const durationMs = Math.round(performance.now() - chunk.startedAt);
  const modelUpdate = policy?.meta.checkpointUpdate ?? 0;
  const flags = reason === "match-end" ? 1 : 0;

  for (let index = 0; index < MAGIC.length; index += 1) {
    view.setUint8(index, MAGIC.charCodeAt(index));
  }

  view.setUint16(8, FORMAT_VERSION, true);
  view.setUint16(10, HEADER_BYTES, true);
  view.setUint16(12, RECORD_BYTES, true);
  view.setUint16(14, SAMPLE_HZ, true);
  view.setUint32(16, sampleCount, true);
  view.setUint32(20, modelUpdate, true);
  view.setUint32(24, durationMs, true);
  view.setUint16(28, game.leftScore, true);
  view.setUint16(30, game.rightScore, true);
  view.setUint16(32, chunk.humanHits, true);
  view.setUint16(34, chunk.botHits, true);
  view.setUint16(36, chunk.maxRally, true);
  view.setUint16(38, chunk.inputChanges, true);
  view.setUint16(40, chunk.visibleSamples, true);
  view.setUint16(42, flags, true);
  view.setUint32(44, chunkIndex, true);

  chunk.samples.forEach((sample, sampleIndex) => {
    const offset = HEADER_BYTES + sampleIndex * RECORD_BYTES;

    sample.humanObservation.forEach((value, index) => {
      view.setInt16(offset + index * 2, quantizeUnit(value), true);
    });
    sample.botObservation.forEach((value, index) => {
      view.setInt16(offset + 18 + index * 2, quantizeUnit(value), true);
    });
    view.setUint8(offset + 36, sample.humanAction);
    view.setUint8(offset + 37, sample.botAction);
    view.setInt16(offset + 38, quantizeReward(sample.humanReward), true);
    view.setInt16(offset + 40, quantizeReward(sample.botReward), true);
    view.setUint8(offset + 42, sample.flags);
    view.setUint8(offset + 43, sample.leftScore);
    view.setUint8(offset + 44, sample.rightScore);
    view.setUint16(offset + 45, clamp(sample.rallyHits, 0, 65535), true);
    view.setUint8(offset + 47, sample.visible ? 1 : 0);
  });

  return buffer;
}

function postBinary(buffer, reason, sampleCount) {
  if (runtimeState.mode === "maintenance") {
    setTrainingMessage(runtimeState.message || "The arcade is cooling down");
    return;
  }

  const headers = {
    "Content-Type": "application/octet-stream",
    "X-Aipong-Match-Id": matchId,
    "X-Aipong-Chunk-Index": String(chunkIndex),
    "X-Aipong-Reason": reason,
  };

  if (reason === "pagehide" && navigator.sendBeacon) {
    const blob = new Blob([buffer], { type: "application/octet-stream" });
    const url = `${LOG_ENDPOINT}?match=${encodeURIComponent(matchId)}&chunk=${chunkIndex}&reason=${encodeURIComponent(reason)}`;
    if (navigator.sendBeacon(url, blob)) {
      setTrainingMessage("Match contribution saved");
      return;
    }
  }

  fetch(LOG_ENDPOINT, {
    method: "POST",
    headers,
    body: buffer,
    keepalive: buffer.byteLength <= 60_000,
  })
    .then((response) => readJsonResponse(response).then((payload) => ({ response, payload })))
    .then(({ response, payload }) => {
      applyRuntimePayload(payload);
      if (!response.ok) {
        throw new Error(`collector returned ${response.status}`);
      }
      setTrainingMessage("Match contribution saved");
    })
    .catch((error) => {
      console.error(error);
      setTrainingMessage("Match contribution could not be saved");
    });
}

function flushChunk(reason) {
  if (chunk.samples.length === 0) {
    return;
  }

  const decision = shouldSendChunk(reason);
  const sampleCount = chunk.samples.length;

  if (decision.ok) {
    const buffer = encodeChunk(reason);
    postBinary(buffer, reason, sampleCount);
  } else {
    setTrainingMessage(`Not sent: ${decision.why}`);
  }

  chunkIndex += 1;
  chunk = createChunk();
}

function stepGame(leftAction, rightAction, envConfig) {
  const result = {
    humanReward: 0,
    botReward: 0,
    flags: 0,
    matchEnded: false,
  };

  game.leftY = movePaddle(game.leftY, leftAction, envConfig);
  game.rightY = movePaddle(game.rightY, rightAction, envConfig);
  game.ballX += game.ballVx * envConfig.dt;
  game.ballY += game.ballVy * envConfig.dt;

  let leftAlignment = 0;
  let rightAlignment = 0;
  const halfHeight = envConfig.height / 2;

  if (game.ballVx < 0) {
    leftAlignment = 1 - clamp(Math.abs(game.ballY - game.leftY) / halfHeight, 0, 1);
  } else if (game.ballVx > 0) {
    rightAlignment = 1 - clamp(Math.abs(game.ballY - game.rightY) / halfHeight, 0, 1);
  }

  const shapingReward = 0.0025 * (leftAlignment - rightAlignment);
  result.humanReward += shapingReward;
  result.botReward -= shapingReward;

  if (game.ballY - envConfig.ball_radius <= 0) {
    game.ballY = envConfig.ball_radius;
    game.ballVy = Math.abs(game.ballVy);
  } else if (game.ballY + envConfig.ball_radius >= envConfig.height) {
    game.ballY = envConfig.height - envConfig.ball_radius;
    game.ballVy = -Math.abs(game.ballVy);
  }

  const leftFront = envConfig.paddle_margin + envConfig.paddle_width;
  const rightFront = envConfig.width - envConfig.paddle_margin - envConfig.paddle_width;
  const halfPaddle = envConfig.paddle_height / 2;

  if (
    game.ballVx < 0 &&
    game.ballX - envConfig.ball_radius <= leftFront &&
    Math.abs(game.ballY - game.leftY) <= halfPaddle
  ) {
    game.ballX = leftFront + envConfig.ball_radius;
    reflectFromPaddle(game, game.leftY, 1, envConfig);
    game.leftHits += 1;
    game.rallyHits += 1;
    game.maxRally = Math.max(game.maxRally, game.rallyHits);
    game.points += POINTS_PER_HIT + game.rallyHits * POINTS_PER_RALLY;
    result.humanReward += envConfig.hit_reward;
    result.botReward -= envConfig.hit_reward;
    result.flags |= 2;
  }

  if (
    game.ballVx > 0 &&
    game.ballX + envConfig.ball_radius >= rightFront &&
    Math.abs(game.ballY - game.rightY) <= halfPaddle
  ) {
    game.ballX = rightFront - envConfig.ball_radius;
    reflectFromPaddle(game, game.rightY, -1, envConfig);
    game.rightHits += 1;
    game.rallyHits += 1;
    game.maxRally = Math.max(game.maxRally, game.rallyHits);
    result.humanReward -= envConfig.hit_reward;
    result.botReward += envConfig.hit_reward;
    result.flags |= 4;
  }

  if (game.ballX < -envConfig.ball_radius) {
    game.rightScore += 1;
    result.humanReward -= 1;
    result.botReward += 1;
    result.flags |= 8;
    serveBall(game, envConfig, -1);
    overlayMessage = "Bot scored";
    overlayUntil = performance.now() + 900;
  } else if (game.ballX > envConfig.width + envConfig.ball_radius) {
    game.leftScore += 1;
    game.points += POINTS_PER_POINT;
    result.humanReward += 1;
    result.botReward -= 1;
    result.flags |= 16;
    serveBall(game, envConfig, 1);
    overlayMessage = "You scored";
    overlayUntil = performance.now() + 900;
  }

  if (game.leftScore >= envConfig.score_to_win || game.rightScore >= envConfig.score_to_win) {
    result.flags |= 1;
    result.matchEnded = true;
  }

  return result;
}

function drawGame(message) {
  context.clearRect(0, 0, config.width, config.height);
  context.fillStyle = "#010503";
  context.fillRect(0, 0, config.width, config.height);

  const fieldGradient = context.createLinearGradient(0, 0, 0, config.height);
  fieldGradient.addColorStop(0, "rgba(49,255,117,0.045)");
  fieldGradient.addColorStop(0.5, "rgba(49,255,117,0.025)");
  fieldGradient.addColorStop(1, "rgba(0,0,0,0.16)");
  context.fillStyle = fieldGradient;
  context.fillRect(0, 0, config.width, config.height);

  context.strokeStyle = "rgba(92, 255, 143, 0.07)";
  context.lineWidth = 1;
  for (let x = 90; x < config.width; x += 90) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, config.height);
    context.stroke();
  }
  for (let y = 75; y < config.height; y += 75) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(config.width, y);
    context.stroke();
  }

  context.strokeStyle = "rgba(167, 255, 191, 0.55)";
  context.lineWidth = 2;
  context.setLineDash([14, 18]);
  context.beginPath();
  context.moveTo(config.width / 2, 0);
  context.lineTo(config.width / 2, config.height);
  context.stroke();
  context.setLineDash([]);

  context.strokeStyle = "rgba(49, 255, 117, 0.28)";
  context.lineWidth = 2;
  context.strokeRect(18, 18, config.width - 36, config.height - 36);

  const paddleGradient = context.createLinearGradient(0, 0, 0, config.height);
  paddleGradient.addColorStop(0, "#e5ffe9");
  paddleGradient.addColorStop(1, "#67ff9a");
  context.fillStyle = paddleGradient;
  context.shadowColor = "rgba(49, 255, 117, 0.38)";
  context.shadowBlur = 18;
  context.beginPath();
  context.roundRect(
    config.paddle_margin,
    game.leftY - config.paddle_height / 2,
    config.paddle_width,
    config.paddle_height,
    4
  );
  context.fill();

  context.beginPath();
  context.roundRect(
    config.width - config.paddle_margin - config.paddle_width,
    game.rightY - config.paddle_height / 2,
    config.paddle_width,
    config.paddle_height,
    4
  );
  context.fill();
  context.shadowBlur = 0;

  context.shadowColor = "rgba(49, 255, 117, 0.78)";
  context.shadowBlur = 24;
  context.fillStyle = "#31ff75";
  context.beginPath();
  context.arc(game.ballX, game.ballY, config.ball_radius, 0, Math.PI * 2);
  context.fill();
  context.shadowBlur = 0;

  context.fillStyle = "rgba(0, 0, 0, 0.42)";
  context.beginPath();
  context.roundRect(config.width / 2 - 92, 20, 184, 42, 8);
  context.fill();

  context.fillStyle = "#e5ffe9";
  context.font = "700 28px Consolas, monospace";
  context.textAlign = "center";
  context.fillText(`${game.leftScore} : ${game.rightScore}`, config.width / 2, 50);

  context.fillStyle = "rgba(198, 233, 205, 0.74)";
  context.font = "15px Consolas, monospace";
  context.textAlign = "left";
  context.fillText("you vs evolving bot", 28, config.height - 46);
  context.fillText(`${formatPoints(game.points)} · hits:${game.leftHits} rally:${game.rallyHits}`, 28, config.height - 24);

  if (message) {
    context.fillStyle = "rgba(1, 5, 3, 0.8)";
    context.fillRect(0, config.height / 2 - 46, config.width, 92);
    context.fillStyle = "#e5ffe9";
    context.font = "700 24px Consolas, monospace";
    context.textAlign = "center";
    context.fillText(message, config.width / 2, config.height / 2 + 8);
  }
}

function resetMatch(startNewSession = true) {
  if (runtimeState.mode === "maintenance") {
    return;
  }

  if (startNewSession) {
    flushChunk("reset");
    matchId = createMatchId();
    chunkIndex = 0;
    chunk = createChunk();
  }

  game = createInitialState(config);
  matchStartedAt = performance.now();
  playRegisteredForMatch = false;
  sampleAccumulator = 0;
  overlayMessage = "New match";
  overlayUntil = performance.now() + 900;
}

function updateHud() {
  const progress = Math.min(100, Math.round((chunk.samples.length / MAX_SAMPLES_PER_CHUNK) * 100));

  scoreText.textContent = formatPoints(game.points);
  if (sampleMeterFill) {
    sampleMeterFill.style.width = `${progress}%`;
  }
  if (sampleCountText) {
    sampleCountText.textContent = `${chunk.samples.length}/${MAX_SAMPLES_PER_CHUNK}`;
  }

  if (performance.now() < transientTrainingUntil) {
    if (trainingStatusText) {
      trainingStatusText.textContent = transientTrainingMessage;
    }
    return;
  }

  if (runtimeState.mode === "maintenance") {
    if (trainingStatusText) {
      trainingStatusText.textContent = runtimeState.message || "The arcade is cooling down";
    }
    return;
  }

  if (runtimeState.mode === "slow") {
    if (trainingStatusText) {
      trainingStatusText.textContent = runtimeState.message || "Saving matches may feel slower for a bit";
    }
    return;
  }

  if (chunk.samples.length > 0) {
    if (trainingStatusText) {
      trainingStatusText.textContent = "Saving this match if it helps the next bot";
    }
  } else {
    if (trainingStatusText) {
      trainingStatusText.textContent = "Ready for a clean match";
    }
  }
}

function tick(time) {
  const frameDelta = Math.min((time - lastTime) / 1000, 0.05);
  lastTime = time;

  if (runtimeState.mode === "maintenance") {
    drawGame("Cooling down. Come back soon.");
    if (time - lastHudUpdate > 250) {
      lastHudUpdate = time;
      updateHud();
    }
    requestAnimationFrame(tick);
    return;
  }

  accumulator += frameDelta;

  while (accumulator >= config.dt) {
    const humanObservation = getLeftObservation(game, config);
    const botObservation = getRightObservation(game, config);
    const humanAction = currentHumanAction();
    const botAction = policy ? choosePolicyAction(policy, botObservation) : chooseTrackerAction(game.ballY, game.rightY);
    const stepResult = stepGame(humanAction, botAction, config);

    sampleAccumulator += config.dt;
    if (sampleAccumulator >= SAMPLE_INTERVAL) {
      addSample(createRecord(humanObservation, botObservation, humanAction, botAction, stepResult));
      sampleAccumulator -= SAMPLE_INTERVAL;
    }

    if (stepResult.matchEnded) {
      const leftWon = game.leftScore > game.rightScore;
      const matchSummary = {
        humanScore: game.leftScore,
        botScore: game.rightScore,
        humanHits: game.leftHits,
        botHits: game.rightHits,
        maxRally: game.maxRally,
        points: game.points,
        durationMs: Math.round(performance.now() - matchStartedAt),
        modelUpdate: policy?.meta.checkpointUpdate ?? 0,
      };
      flushChunk("match-end");
      submitLeaderboardEntry(matchSummary);
      matchId = createMatchId();
      chunkIndex = 0;
      chunk = createChunk();
      resetMatch(false);
      overlayMessage = leftWon ? "You won. New match." : "Bot won. New match.";
      overlayUntil = time + 1400;
    }

    accumulator -= config.dt;
  }

  drawGame(time < overlayUntil ? overlayMessage : "");

  if (time - lastHudUpdate > 250) {
    lastHudUpdate = time;
    updateHud();
  }

  requestAnimationFrame(tick);
}

function updatePointerTarget(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleY = config.height / rect.height;
  pointerTarget = (event.clientY - rect.top) * scaleY;
}

function handlePointerDown(event) {
  registerPlay();
  pointerActive = true;
  canvas.focus();
  canvas.setPointerCapture(event.pointerId);
  updatePointerTarget(event);
}

function handlePointerMove(event) {
  if (pointerActive) {
    updatePointerTarget(event);
  }
}

function stopPointerControl(event) {
  pointerActive = false;
  pointerTarget = null;
  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
}

window.addEventListener("keydown", (event) => {
  if (event.code === "KeyW" || event.code === "ArrowUp") {
    registerPlay();
    keyState.up = true;
    event.preventDefault();
  }
  if (event.code === "KeyS" || event.code === "ArrowDown") {
    registerPlay();
    keyState.down = true;
    event.preventDefault();
  }
  if (event.code === "Space") {
    registerPlay();
    serveBall(game, config);
    event.preventDefault();
  }
});

window.addEventListener("keyup", (event) => {
  if (event.code === "KeyW" || event.code === "ArrowUp") {
    keyState.up = false;
  }
  if (event.code === "KeyS" || event.code === "ArrowDown") {
    keyState.down = false;
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") {
    pointerActive = false;
    pointerTarget = null;
  }
});

window.addEventListener("pagehide", () => {
  flushChunk("pagehide");
});

canvas.addEventListener("pointerdown", handlePointerDown);
canvas.addEventListener("pointermove", handlePointerMove);
canvas.addEventListener("pointerup", stopPointerControl);
canvas.addEventListener("pointercancel", stopPointerControl);
canvas.addEventListener("pointerleave", () => {
  pointerActive = false;
  pointerTarget = null;
});
resetButton.addEventListener("click", () => resetMatch(true));
playerForm.addEventListener("submit", (event) => {
  event.preventDefault();
  setPlayerName(playerNameInput.value);
  canvas.focus();
});
leaderboardRefreshButton.addEventListener("click", loadLeaderboard);

initializePlayerName();
renderRuntimeState();
loadLeaderboard();
loadPlayStats();
setInterval(loadPlayStats, STATS_REFRESH_MS);

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    loadPlayStats();
  }
});

fetch(MODEL_URL)
  .then((response) => {
    if (!response.ok) {
      throw new Error(`Failed to load model: ${response.status}`);
    }

    return response.json();
  })
  .then((model) => {
    policy = compilePolicy(model);
    config = model.envConfig;
    canvas.width = config.width;
    canvas.height = config.height;
    game = createInitialState(config);
    statusText.textContent = `Generation ${model.checkpointUpdate}`;
    overlayMessage = `Generation ${model.checkpointUpdate} online`;
    overlayUntil = performance.now() + 2200;
  })
  .catch((error) => {
    console.error(error);
    statusText.textContent = "Bot failed to load; using tracker fallback";
    overlayMessage = "Bot failed to load. Using tracker fallback.";
    overlayUntil = performance.now() + 2600;
  });

drawGame(overlayMessage);
requestAnimationFrame(tick);
