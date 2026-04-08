CREATE TABLE IF NOT EXISTS play_totals (
  key TEXT PRIMARY KEY,
  plays INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS played_matches (
  match_key TEXT PRIMARY KEY,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS leaderboard_entries (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  name TEXT NOT NULL,
  human_score INTEGER NOT NULL,
  bot_score INTEGER NOT NULL,
  points INTEGER NOT NULL,
  margin INTEGER NOT NULL,
  human_won INTEGER NOT NULL,
  human_hits INTEGER NOT NULL,
  bot_hits INTEGER NOT NULL,
  max_rally INTEGER NOT NULL,
  duration_ms INTEGER NOT NULL,
  model_update INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS leaderboard_rank_idx
ON leaderboard_entries (
  points DESC,
  human_won DESC,
  human_score DESC,
  margin DESC,
  max_rally DESC,
  human_hits DESC,
  created_at ASC
);

CREATE TABLE IF NOT EXISTS usage_state (
  window_key TEXT PRIMARY KEY,
  window TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  used_units INTEGER NOT NULL DEFAULT 0,
  trajectory_uploads INTEGER NOT NULL DEFAULT 0,
  trajectory_bytes INTEGER NOT NULL DEFAULT 0,
  leaderboard_posts INTEGER NOT NULL DEFAULT 0,
  play_registrations INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trajectory_manifest (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  object_key TEXT NOT NULL,
  bytes INTEGER NOT NULL,
  reason TEXT NOT NULL,
  match_id TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  checkpoint_update INTEGER NOT NULL,
  sample_count INTEGER NOT NULL,
  duration_ms INTEGER NOT NULL
);
