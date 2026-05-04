# Web image: Next.js standalone build of the AG-UI/CopilotKit client.
FROM node:20-alpine AS deps
WORKDIR /app
COPY web/package.json web/package-lock.json* ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi

FROM node:20-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY web/ ./
RUN npm run build
RUN mkdir -p ./public

FROM node:20-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production HOSTNAME=0.0.0.0 PORT=3000
RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 nextjs
COPY --from=build --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=build --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=build --chown=nextjs:nodejs /app/public ./public
USER nextjs
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=3s CMD wget -q -O- http://localhost:3000/ >/dev/null || exit 1
ENTRYPOINT ["node", "server.js"]
