#!/usr/bin/env bash
# Wait for the Kalibr image build, then run camera intrinsic calibration (OpenVINS guide step 2).
exec > /root/navlori/logs/kalibr_cam.log 2>&1
set -x
export DOCKER_CONFIG=/root/.docker_plain
until grep -q KALIBR_BUILD_EXIT /root/navlori/logs/kalibr_build.log; do sleep 30; done
grep KALIBR_BUILD_EXIT /root/navlori/logs/kalibr_build.log
docker image inspect kalibr >/dev/null 2>&1 || { echo "NO_KALIBR_IMAGE"; exit 1; }
docker run --rm -v /root/navlori/kalibr_work:/data -w /data -e MPLBACKEND=Agg \
  --entrypoint /bin/bash kalibr -c "source /catkin_ws/devel/setup.bash && \
  rosrun kalibr kalibr_calibrate_cameras --bag /data/calib_cam.bag \
  --target /data/aprilgrid_8x6_30mm.yaml --models pinhole-radtan \
  --topics /camera/image_raw --bag-freq 4.0 --dont-show-report"
echo "KALIBR_CAM_EXIT=$?"
ls -l /root/navlori/kalibr_work
