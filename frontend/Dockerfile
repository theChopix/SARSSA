# syntax=docker/dockerfile:1
#
# Frontend image: build the Vite/React app, then serve the static bundle
# with nginx.
#
# NOTE: `src/constants.ts` hardcodes the backend URL to
# `http://localhost:8000` and the backend's CORS allow-list is
# `http://localhost:5173`. Compose publishes this container on host port
# 5173 and the backend on 8000, so the browser (running on the host)
# reaches both without any code change. Open the UI at
# http://localhost:5173.

FROM node:22-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:1.27-alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
