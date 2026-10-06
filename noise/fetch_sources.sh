#!/usr/bin/env bash
# Скачать исходники шумов в noise/sources (см. SOURCES.md), затем:
#   python uniquify.py pack
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p sources

MIXKIT=(
  broken-static-on-an-old-tv-25350
  dark-noise-on-a-tv-25349
  white-static-on-a-tv-25348
  dust-floating-on-black-background-47285
  film-grain-close-up-26657
  golden-particles-background-46316
  old-film-grain-playing-on-a-surface-26677
  old-film-playing-on-a-white-background-26676
  screen-with-static-and-pixel-noise-4111
  static-of-a-television-4307
  television-glich-texture-3524
  television-screen-with-static-in-black-and-white-4135
  tv-static-on-an-old-tv-screen-in-the-dark-46290
  vintage-film-grain-26656
)
for slug in "${MIXKIT[@]}"; do
  id="${slug##*-}"
  out="sources/mixkit_${slug}.mp4"
  [ -s "$out" ] && continue
  # 1080p есть не у всех — тогда берём 720p.
  curl -sfL -o "$out" "https://assets.mixkit.co/videos/$id/$id-1080.mp4" \
    || curl -sfL -o "$out" "https://assets.mixkit.co/videos/$id/$id-720.mp4"
  echo "ok $out"
done

UA="uniquifier/1.0 (noise fetch)"
fetch_commons() {
  [ -s "sources/$1" ] && return
  curl -sfL -A "$UA" -o "sources/$1" "$2" && echo "ok sources/$1"
}
fetch_commons commons_static_white_noise.webm \
  "https://upload.wikimedia.org/wikipedia/commons/e/ea/Static_with_white_noise.webm"
fetch_commons commons_vhs_static.webm \
  "https://upload.wikimedia.org/wikipedia/commons/b/b3/FREE_real_VHS_static.webm"

ls sources | wc -l | xargs echo "исходников:"
