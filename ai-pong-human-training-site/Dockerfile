FROM node:22-alpine

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=8787

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY collector ./collector
COPY web ./web
COPY docs ./docs
COPY scripts ./scripts

RUN mkdir -p data

EXPOSE 8787

CMD ["node", "collector/server.mjs"]
