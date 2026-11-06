""" 
                     
                              [ RealSense D435i ]
                                      |
           ---------------------------------------------------------
          |                                                         |
[ /camera/color/image_raw (RGB) ]                 [ /camera/depth/image_rect_raw (depth) ]
          |                                                         |
          ----------------------                                    |
          |                     |                                   |
          v                     v                                   v
[ camera_scan_node ]       [ lap_counter_node]              [ depth_scan_node ]
  |_ camera_model.py        similarity threshold              |_ depth.py
  |_ masks.py                                                       |
  |_ boundary.py                                                    |
  |_ debug_draw.py                                                  |
                                                                    |
                    |                                               |
                    v                                               |
        ------------------------                                    |
       |                        |                                   |
       v                        v                                   v
[ /camera_scan ]        [ /camera_scan/debug ]                 [ /depth_scan ]       
  |_lane_follow_node      |_rqt_image_view
  |_gap_follow_node                                                 |
  |_wall_follow_node           |                                    |
                               |                                    |
        |                      |                                    |
[ LaserScan ]          [ sensor_msgs/Image ]                 [ LaserScan ]
        |                      |                                    |
        |                      |                                    |
        | steering + speed     | neg speed after N laps             | ranges + angle
        |                      |                                    | 
        v                      v                                    |
        ------------------------                                    | 
[ /drive_request ]                                                  |
AckermannDriveStamped                                               |
        |-----------------------                                    |
        |                      |                                    |
        v                      v                                    v

[ safety_node ]           [ safety_node ]                      [ safety_node ]
               
       |-----------------------------------------------------------------                              
       |                         
    [ /drive] 
    AckermannDriveStamped                 
       |
       v
     [ car ]  ------------------> [ /odom ] -------------> [safety_node] 
                                   Odometry
 """
