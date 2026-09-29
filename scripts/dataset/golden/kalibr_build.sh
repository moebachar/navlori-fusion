#!/usr/bin/env bash
# Build Kalibr's official Dockerfile_ros1_20_04 on the native WSL Docker engine.
exec > /root/navlori/logs/kalibr_build.log 2>&1
set -x
export DOCKER_CONFIG=/root/.docker_plain; mkdir -p "$DOCKER_CONFIG"; echo '{}' > "$DOCKER_CONFIG/config.json"
if ! timeout 10 docker version >/dev/null 2>&1; then
  rm -f /var/run/docker.sock /var/run/docker.pid
  (dockerd > /root/navlori/logs/dockerd.log 2>&1 &)
  for i in $(seq 30); do timeout 10 docker version >/dev/null 2>&1 && break; sleep 2; done
fi
docker version --format 'engine {{.Server.Version}}'
cd /mnt/x/kalibr && docker build -t kalibr -f Dockerfile_ros1_20_04 .
echo "KALIBR_BUILD_EXIT=$?"
