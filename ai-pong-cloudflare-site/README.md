# AI Pong Cloudflare Site

This is the Cloudflare-native deployable package for the AI Pong game.

It includes:

- static frontend assets in [`public`](G:/Projects/AI%20PONG/ai-pong-cloudflare-site/public)
- a Worker backend in [`src/worker.js`](G:/Projects/AI%20PONG/ai-pong-cloudflare-site/src/worker.js)
- D1 schema in [`schema.sql`](G:/Projects/AI%20PONG/ai-pong-cloudflare-site/schema.sql)
- Cloudflare config in [`wrangler.jsonc`](G:/Projects/AI%20PONG/ai-pong-cloudflare-site/wrangler.jsonc)

## What Lives Here

- game frontend
- `/api/ai-pong/runtime`
- `/api/ai-pong/stats`
- `/api/ai-pong/leaderboard`
- `/api/ai-pong/trajectory`
- R2 trajectory storage
- D1 leaderboard, play count, and usage guard state

## Setup

1. Create an R2 bucket.
2. Create a D1 database.
3. Replace the placeholder bucket name and D1 database id in [`wrangler.jsonc`](G:/Projects/AI%20PONG/ai-pong-cloudflare-site/wrangler.jsonc).
4. Set `ALLOWED_ORIGIN` to your real domain.
5. Apply the D1 migration:

```powershell
cd "G:\Projects\AI PONG\ai-pong-cloudflare-site"
npm install
npx wrangler d1 migrations apply ai-pong --remote
```

6. Verify the package:

```powershell
npm run verify
```

7. Deploy:

```powershell
npm run deploy
```

## Cloudflare Build Settings

Use these in the dashboard:

- Build command: leave blank
- Deploy command: `npx wrangler deploy`
- Non-production branch deploy command: `npx wrangler versions upload`
- Path: `ai-pong-cloudflare-site`

## One-Time Cloudflare Checklist

1. In `Workers & Pages`, create a Worker from your repo and set the root path to `ai-pong-cloudflare-site`.
2. In `Storage & Databases`, create:
   - one D1 database, for example `ai-pong`
   - one private R2 bucket for trajectories
3. Update [`wrangler.jsonc`](G:/Projects/AI%20PONG/ai-pong-cloudflare-site/wrangler.jsonc) with the real D1 database id and R2 bucket name.
4. In the Worker settings, override `ALLOWED_ORIGIN` with your production URL.
5. Run the D1 migration command once.
6. Deploy from the Cloudflare dashboard or by running `npm run deploy`.

## What To Point Cloudflare At

This directory is the deployable package:

- repo path: [`ai-pong-cloudflare-site`](G:/Projects/AI%20PONG/ai-pong-cloudflare-site)
- frontend assets: [`public`](G:/Projects/AI%20PONG/ai-pong-cloudflare-site/public)
- backend Worker: [`src/worker.js`](G:/Projects/AI%20PONG/ai-pong-cloudflare-site/src/worker.js)

## Notes

- This package is the deployable Cloudflare app.
- The laptop retraining loop still lives in [`ai-pong-training-runner`](G:/Projects/AI%20PONG/ai-pong-training-runner).
- The Python training code still lives in [`ai_pong`](G:/Projects/AI%20PONG/ai_pong).
