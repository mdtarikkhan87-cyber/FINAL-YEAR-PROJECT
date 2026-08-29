# Next.js frontend, production build.
#
#   docker build -f web/Dockerfile -t perm-web ./web
#
# Build context is web/. Nothing outside it is needed.
#
# Deliberately no NEXT_PUBLIC_API_BASE build argument: that value would be
# inlined into the client bundle and freeze the API URL into the image. The
# browser instead calls the same-origin /api route, which the running server
# forwards to API_INTERNAL_URL at request time. See app/api/[...path]/route.ts.

# ---------------------------------------------------------------- deps -------
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
# `npm ci` installs exactly the lockfile, which is what makes the image
# reproducible rather than "whatever was latest on build day".
RUN npm ci

# --------------------------------------------------------------- builder -----
FROM node:22-alpine AS builder
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# ---------------------------------------------------------------- runner -----
FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN addgroup -g 1001 -S nodejs \
    && adduser -S -u 1001 -G nodejs nextjs

# `output: "standalone"` produces a server.js plus only the node_modules
# actually reached at runtime; the full dependency tree never enters this stage.
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD wget --quiet --tries=1 --spider http://127.0.0.1:3000/ || exit 1

CMD ["node", "server.js"]
