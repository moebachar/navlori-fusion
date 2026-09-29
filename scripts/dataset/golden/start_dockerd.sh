#!/bin/bash
exec > /var/log/dockerd.log 2>&1
rm -f /var/run/docker.sock /var/run/docker.pid
exec dockerd
