import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import math

class UltimateGapFollower(Node):
    def __init__(self):
        super().__init__('ultimate_gap_follower_node')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        
        # ==========================================
        # 🏎️ Parameters จูนพิเศษ (เพิ่มระยะข้างรถ)
        # ==========================================
        # [สำคัญ] เพิ่มความกว้างรถเป็น 0.22 เมตร (22 ซม.) เพื่อบังคับให้หุ่นวิ่งห่างจากกำแพงด้านข้างมากขึ้น
        self.ROBOT_RADIUS = 0.22      
        self.MAX_RANGE = 3.0          
        # ลดเกณฑ์ลงมาเหลือ 0.2 เพื่อให้หุ่นจับขอบกำแพงและเริ่มตีวงเลี้ยวได้ไวขึ้น
        self.DISPARITY_THRESH = 0.2   
        self.prev_error = 0.0
        
        self.get_logger().info('🚀 ULTIMATE V.4: โหมดตีโค้งกว้าง เผื่อระยะข้างรถ ไม่ชนกำแพงแน่นอน ลุย!')

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angle_inc = msg.angle_increment
        
        # ==========================================
        # 0. กรอง Noise ของเซนเซอร์ LiDAR LDS-01
        # ==========================================
        # เตะค่า Error ของ LiDAR (0.0 หรือใกล้เกินไป) ทิ้ง เพื่อไม่ให้รถเบรกเองฟรีๆ
        ranges = np.where(np.isinf(ranges) | np.isnan(ranges) | (ranges < 0.12), self.MAX_RANGE, ranges)
        ranges = np.clip(ranges, 0.0, self.MAX_RANGE)

        # ตัดข้อมูลเอาเฉพาะ 180 องศาด้านหน้า
        right_fov = ranges[270:360]
        left_fov = ranges[0:91]
        proc_ranges = np.concatenate((right_fov, left_fov))
        CENTER_IDX = 90

        # ==========================================
        # 1. Disparity Extender (พองขนาดสิ่งกีดขวาง)
        # ==========================================
        disparities = np.diff(proc_ranges)
        disparity_indices = np.where(np.abs(disparities) > self.DISPARITY_THRESH)[0]

        for idx in disparity_indices:
            dist1 = proc_ranges[idx]
            dist2 = proc_ranges[idx + 1]
            closer_dist = min(dist1, dist2)
            
            if closer_dist > 0.1:
                # ล็อกมุมขยายสูงสุด 45 องศา (sin 45 = 0.707)
                extend_angle = math.asin(min(self.ROBOT_RADIUS / closer_dist, 0.707)) 
                extend_idx_count = int(extend_angle / angle_inc)
            else:
                extend_idx_count = 15

            if dist1 < dist2:
                start_e = idx + 1
                end_e = min(len(proc_ranges), idx + 1 + extend_idx_count)
                proc_ranges[start_e:end_e] = np.minimum(proc_ranges[start_e:end_e], closer_dist)
            else:
                start_e = max(0, idx - extend_idx_count)
                end_e = idx + 1
                proc_ranges[start_e:end_e] = np.minimum(proc_ranges[start_e:end_e], closer_dist)

        # ==========================================
        # 2. Safety Bubble (เกราะกันชนส่วนตัว)
        # ==========================================
        min_idx = np.argmin(proc_ranges)
        min_dist = proc_ranges[min_idx]
        
        if min_dist < self.MAX_RANGE:
            bubble_angle = math.asin(min(self.ROBOT_RADIUS / max(min_dist, 0.1), 0.707))
            num_indices = int(bubble_angle / angle_inc)
            
            start_b = max(0, min_idx - num_indices)
            end_b = min(len(proc_ranges) - 1, min_idx + num_indices)
            proc_ranges[start_b:end_b+1] = 0.0

        # ==========================================
        # 3. Extract Max Gap (หาช่องว่างที่ใหญ่ที่สุด)
        # ==========================================
        non_zeros = np.where(proc_ranges > 0.1)[0]
        if len(non_zeros) == 0:
            # Failsafe: โดนบีบติดกำแพง ถอยหลังนิดนึงพร้อมหักพวงมาลัย
            self.publish_cmd(-0.05, 0.5) 
            return
            
        gaps = np.split(non_zeros, np.where(np.diff(non_zeros) != 1)[0] + 1)
        largest_gap = max(gaps, key=len)

        # ==========================================
        # 4. AI Cost Function (หาเป้าหมายที่ดีที่สุด)
        # ==========================================
        best_score = -float('inf')
        best_idx = CENTER_IDX

        for idx in largest_gap:
            depth = proc_ranges[idx]
            # ลดค่าบทลงโทษลง (0.01) เพื่อให้รถกล้าที่จะหักเลี้ยวหลบกำแพงมากขึ้น ไม่ดื้อดึงจะวิ่งตรง
            turn_penalty = abs(idx - CENTER_IDX) * 0.01 
            score = depth - turn_penalty
            if score > best_score:
                best_score = score
                best_idx = idx

        target_angle_rad = (best_idx - CENTER_IDX) * angle_inc

        # ==========================================
        # 5. Advanced PD Control & Exponential Speed
        # ==========================================
        # เพิ่มความไวพวงมาลัย ให้กระชากหลบได้ทันเวลาเจอของข้างหน้า
        Kp = 1.5
        Kd = 0.5
        error = target_angle_rad
        steering = (Kp * error) + (Kd * (error - self.prev_error))
        self.prev_error = error
        
        # ปลดล็อกคอพวงมาลัยให้หักเลี้ยวได้เยอะขึ้น (2.0 rad/s)
        steering = max(-2.0, min(2.0, steering))

        # สปีดรถ: วิ่งตรงทำเวลาได้ 0.20 ทางโค้งหักศอกชะลอเหลือ 0.05
        MAX_SPEED = 0.20
        MIN_SPEED = 0.05
        speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * math.exp(-3.5 * abs(steering))

        self.publish_cmd(speed, steering)

    def publish_cmd(self, speed, steering):
        twist = Twist()
        twist.linear.x = float(speed)
        twist.angular.z = float(steering)
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = UltimateGapFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.publish_cmd(0.0, 0.0)
        node.get_logger().info('🛑 เบรกฉุกเฉิน! ปิดระบบอย่างปลอดภัย')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()