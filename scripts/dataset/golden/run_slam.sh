#!/usr/bin/env bash
exec > /root/navlori/logs/slam.log 2>&1
source /root/navlori/venv/bin/activate
sed 's|^ROOT = .*|ROOT = "/root/navlori/incoming/exported"|' \
  /mnt/x/side_navlori/data/scripts/build_ground_truth.py > /root/build_gt.py
echo "=== running lidar SLAM (PLICP scan-to-map + refinement) ==="
python /root/build_gt.py
echo "SLAM_EXIT rc=$?"
cp /root/navlori/incoming/exported/ground_truth/gt_overview.png /mnt/x/side_navlori/_inspect/ 2>/dev/null && echo "copied gt_overview.png -> _inspect"
cp /root/navlori/incoming/exported/ground_truth/map_preview.png /mnt/x/side_navlori/_inspect/ 2>/dev/null && echo "copied map_preview.png -> _inspect"
