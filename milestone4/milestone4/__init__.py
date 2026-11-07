"""
CPEN 391 Milestone 4: LiDAR-primary autonomous driving with camera fusion.

Nodes (one ROS 2 executable each):
    camera_scan_node    colour image -> /camera_scan and /camera_box_scan
    depth_scan_node     depth image  -> /depth_scan (floor rejected)
    fusion_node         /scan + camera scans -> /fused_scan and /obstacle_scan
    gap_follow_node     disparity extender on /fused_scan -> /drive_request
    safety_node         path-aware AEB and watchdog, the only node publishing /drive
    lap_counter_node    camera + LiDAR + odometry lap counting -> latched stop

Helper modules (no ROS code):
    camera_model, masks, boundary, debug_draw, depth, fusion
"""
