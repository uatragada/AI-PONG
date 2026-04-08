# AI Pong Human Training Site

Standalone website and collector for learning from human Pong players.

This is intentionally separate from the portfolio site. It serves a browser game, runs the
current bot model locally, records compact binary trajectory chunks, filters low-quality games
before upload, validates accepted `.aipong` files on the server, and stores them for laptop-side
training.

## Run Locally

```bash
cd "G:\Projects\AI PONG\ai-pong-human-training-site"
npm start
```

Open:

```text
http://localhost:8787
```

Health check:

```text
http://localhost:8787/api/ai-pong/health
```

The health endpoint only exposes public-safe status by default. Set
`AIPONG_EXPOSE_HEALTH_DETAILS=1` temporarily if you need local path and limit diagnostics.

Optional local env setup:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` or set environment variables in your shell/host dashboard.

## Data Flow

1. A visitor plays against the current right-side bot.
2. The browser samples human and bot observations/actions at 10 Hz.
3. The browser waits until the match ends or one minute passes.
4. Before uploading, the browser filters out bad chunks:
   - fewer than 120 samples
   - shorter than 20 seconds unless it was a full minute chunk
   - fewer than 4 human action changes
   - no human paddle hit
   - tab hidden for too much of the chunk
   - bot model failed to load
5. Good chunks are POSTed as `application/octet-stream` to `/api/ai-pong/trajectory`.
6. The collector validates the binary header, recomputes server-side quality stats, rejects malformed or low-quality uploads, and writes accepted files under `data/trajectories/`.
7. A tiny `manifest.jsonl` records where each binary file landed.

The raw training trajectories are binary, not JSON. The manifest is only operational metadata.

The site also keeps a small match leaderboard at:

```text
data/leaderboard.json
```

Leaderboard entries are JSON summaries only: player name, score, hit counts, max rally,
duration, and model update. They are not used as the supervised training trajectory source.

## Pulling Data From Another PC

By default data lands here:

```text
G:\Projects\AI PONG\ai-pong-human-training-site\data
```

You can copy or sync that directory to the training machine with your preferred tool.
Examples:

```bash
robocopy "\\COLLECTOR-PC\ai-pong-data" "D:\training-data\ai-pong" /MIR
```

or:

```bash
rsync -av collector:/path/to/ai-pong-human-training-site/data/ ./data/
```

To store data elsewhere on the collector:

```bash
$env:AIPONG_DATA_DIR = "E:\ai-pong-training-data"
npm start
```

## Scheduled Training Runner

The laptop-side scheduled workflow lives next to this site:

```text
G:\Projects\AI PONG\ai-pong-training-runner
```

Use that runner for the real train, evaluate, and promote loop. This website now loads:

```text
web/models/current-policy.json
```

so a passing scheduled training run can replace one model artifact without editing the game.

## Train A Starter Supervised Model

After pulling collected `.aipong` files onto a training machine:

```bash
python scripts/train_supervised_from_aipong.py data/trajectories --epochs 10 --output web/models/human-supervised-policy.json
```

This is a starter behavior-cloning loop. It trains from:

```text
input:  human_observation
label:  human_action
```

Because the observations are side-relative, the learned policy can be run on the bot side by
feeding it the bot's own mirrored observation at inference time.

## Decode For Supervised Learning

```bash
python scripts/decode_aipong.py data/trajectories --out data/training_samples.jsonl
```

For supervised learning:

- `human_observation` is the input.
- `human_action` is the label.
- `bot_observation`, `bot_action`, and rewards are included for filtering and weighting.

See `docs/binary-format.md` for the binary layout.

## Deployment Notes

This app is deployable as a single persistent Node service now. That can be a home server,
small VPS, Fly.io-style machine, Render/Railway persistent service, or anything else that
keeps local files between restarts.

Before public launch:

```powershell
$env:NODE_ENV = "production"
$env:AIPONG_ALLOWED_ORIGIN = "https://your-real-domain.com"
$env:AIPONG_STORAGE = "local"
$env:AIPONG_DATA_DIR = "E:\ai-pong-training-data"
$env:AIPONG_TRUST_PROXY_HEADERS = "1" # only when behind Cloudflare or another trusted proxy
npm start
```

Or build the included container:

```powershell
docker build -t ai-pong-human-training-site .
docker run --rm -p 8787:8787 -v ${PWD}\data:/app/data --env-file .env ai-pong-human-training-site
```

Use a persistent `AIPONG_DATA_DIR`. The collector stores:

```text
trajectories: data/trajectories/
manifest:     data/manifest.jsonl
leaderboard:  data/leaderboard.json
play count:   data/play-stats.json
usage guard:  data/usage-budget.json
```

The API now includes:

- request size caps for trajectory uploads
- server-side binary format and quality validation
- per-IP rate limits for trajectory, leaderboard, and play-count posts
- duplicate play-count protection per match id
- optional usage-budget slowdown and maintenance mode
- basic security headers and a same-origin CSP

Still put Cloudflare in front of the app for bot/rate protection before sharing it widely.
If you deploy to a serverless host with an ephemeral filesystem, switch trajectory storage to
Cloudflare R2 and move leaderboard/play-count state to a durable store too.

Before a real launch, clear or archive dev-only local data if you do not want test plays on
the public board:

```powershell
Remove-Item data\play-stats.json -ErrorAction SilentlyContinue
Remove-Item data\leaderboard.json -ErrorAction SilentlyContinue
```

## Usage Budget Guardrail

You can give the site a simple usage budget and let it protect itself:

- below the slowdown threshold: normal
- above the slowdown threshold: write requests are artificially delayed
- at the maintenance threshold: the home page switches to `maintenance.html` and gameplay write APIs return `503`

Starter example:

```powershell
$env:AIPONG_USAGE_BUDGET_UNITS = "10000"
$env:AIPONG_USAGE_WINDOW = "monthly"
$env:AIPONG_USAGE_SLOWDOWN_RATIO = "0.65"
$env:AIPONG_USAGE_MAINTENANCE_RATIO = "0.8"
```

The current defaults score usage like this:

- play registration: `1` unit
- leaderboard post: `1` unit
- trajectory upload: `1 + ceil(bytes / AIPONG_USAGE_BYTES_PER_UNIT)` units

With the default `AIPONG_USAGE_BYTES_PER_UNIT=25000`, a typical 1-minute chunk costs about `3` units.

You can inspect the live budget state at:

```text
/api/ai-pong/runtime
```

## Cloudflare R2 Storage

The collector can write accepted `.aipong` chunks directly to Cloudflare R2. The browser
does not change; it still posts compact binary chunks to `/api/ai-pong/trajectory`.

R2 is optional for now. You can do it later by setting the environment variables below.
Leaderboard and play-count JSON still use the local `AIPONG_DATA_DIR`, so keep those on a
persistent host or move them to a database before using a stateless serverless deployment.

Install dependencies once:

```bash
npm install
```

Set these environment variables before starting the collector:

```powershell
$env:AIPONG_STORAGE = "r2"
$env:AIPONG_R2_ACCOUNT_ID = "your-cloudflare-account-id"
$env:AIPONG_R2_BUCKET = "your-r2-bucket-name"
$env:AIPONG_R2_ACCESS_KEY_ID = "your-r2-access-key-id"
$env:AIPONG_R2_SECRET_ACCESS_KEY = "your-r2-secret-access-key"
$env:AIPONG_R2_PREFIX = "ai-pong-training"
npm start
```

Uploaded objects land under:

```text
ai-pong-training/trajectories/checkpoint-<model-update>/YYYY-MM-DD/*.aipong
ai-pong-training/manifests/checkpoint-<model-update>/YYYY-MM-DD/*.json
```

From a different local training PC, configure the AWS CLI with the same R2 access keys and
sync the binary chunks down:

```powershell
aws s3 sync s3://your-r2-bucket-name/ai-pong-training/trajectories ./data/trajectories --endpoint-url https://your-cloudflare-account-id.r2.cloudflarestorage.com
```

Then train from the synced files:

```powershell
python scripts/train_supervised_from_aipong.py data/trajectories --epochs 10 --output web/models/human-supervised-policy.json
```
