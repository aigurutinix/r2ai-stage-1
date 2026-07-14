FROM node:20-alpine AS dependencies
WORKDIR /app/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci

FROM node:20-alpine AS builder
WORKDIR /app/ui
COPY --from=dependencies /app/ui/node_modules ./node_modules
COPY ui ./
COPY backend/config.yaml /app/backend/config.yaml
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app/ui
ENV NODE_ENV=production \
    PORT=5173 \
    HOSTNAME=0.0.0.0 \
    BACKEND_CONFIG_PATH=/app/backend/config.yaml

RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 nextjs

COPY --from=builder /app/ui/.next/standalone ./
COPY --from=builder /app/ui/.next/static ./.next/static
COPY --from=builder /app/backend/config.yaml /app/backend/config.yaml

USER nextjs
CMD ["node", "server.js"]
