import rclpy
from rclpy.node import Node
import math

# Messages
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import Imu

# นำเข้า Message Type สำหรับรับค่า Encoder จากล้อหุ่นยนต์
from turtlebot3_msgs.msg import SensorState 

from tf2_ros import TransformBroadcaster

class TurtlebotOdomNode(Node):
    def __init__(self):
        super().__init__('turtlebot_odom_node')

        # 1. Subscribers
        # รับค่า Ticks จากล้อหุ่นยนต์
        self.sensor_sub = self.create_subscription(SensorState, '/sensor_state', self.sensor_callback, 10)
        # รับค่ามุมจาก IMU (เอาไว้ช่วยชดเชยตอนเลี้ยวให้แม่นขึ้นไปอีก)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        
        # 2. Publishers
        self.odom_pub = self.create_publisher(Odometry, '/custom_odom', 10)
        self.path_pub = self.create_publisher(Path, '/robot_path', 10)

        # 3. TF Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # 4. ข้อมูลพื้นฐาน (Constants) ตามสไลด์อาจารย์
        self.L = 0.160              # Wheel Base (เมตร)
        self.resolution = 4096.0    # Encoder Resolution (ticks/rev)
        self.r = 0.033              # รัศมีล้อ Turtlebot3 มาตรฐาน (เมตร)

        # State Variables สำหรับคำนวณตำแหน่ง
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # ตัวแปรเก็บค่าความเร็ว
        self.v = 0.0              
        self.theta_dot = 0.0      
        self.imu_yaw_rate = 0.0
        self.imu_active = False

        # ตัวแปรเก็บค่า Ticks รอบก่อนหน้า (Step 1)
        self.left_encoder_prev = None
        self.right_encoder_prev = None
        self.last_sensor_time = None

        # Path message
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'

        # 5. Timer (วนลูปอัปเดตตำแหน่ง 50Hz)
        timer_period = 1.0 / 50.0  
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.last_timer_time = self.get_clock().now()

        self.get_logger().info('Odom Node is running with ENCODER Kinematics!')

    def imu_callback(self, msg):
        self.imu_yaw_rate = msg.angular_velocity.z
        self.imu_active = True

    def sensor_callback(self, msg):
        current_time = self.get_clock().now()

        # เช็กว่าเป็นการรับค่าครั้งแรกหรือไม่ ถ้าใช่ให้เก็บค่าเริ่มต้นไว้ก่อน
        if self.left_encoder_prev is None:
            self.left_encoder_prev = msg.left_encoder
            self.right_encoder_prev = msg.right_encoder
            self.last_sensor_time = current_time
            return

        # คำนวณหา dt (ระยะเวลาที่ผ่านไปตั้งแต่รับค่า sensor ครั้งล่าสุด)
        dt = (current_time - self.last_sensor_time).nanoseconds / 1e9
        if dt <= 0:
            return

        # --- Step 1: หาความแตกต่างของ Ticks ---
        delta_ticks_L = msg.left_encoder - self.left_encoder_prev
        delta_ticks_R = msg.right_encoder - self.right_encoder_prev

        # --- Step 2: แปลง Ticks เป็นระยะทางขยับ (ds) ---
        ds_L = (2 * math.pi * self.r * delta_ticks_L) / self.resolution
        ds_R = (2 * math.pi * self.r * delta_ticks_R) / self.resolution

        # --- Step 3: คำนวณหา v และ omega ---
        self.v = (ds_R + ds_L) / 2.0 / dt
        self.theta_dot = (ds_R - ds_L) / self.L / dt

        # อัปเดตค่า Ticks และเวลา ไว้ใช้ในลูปรอบถัดไป
        self.left_encoder_prev = msg.left_encoder
        self.right_encoder_prev = msg.right_encoder
        self.last_sensor_time = current_time

    def timer_callback(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_timer_time).nanoseconds / 1e9
        self.last_timer_time = current_time

        # ผสานข้อมูล (Sensor Fusion): ใช้ความเร็วเส้นตรงจากล้อ + ใช้ความเร็วหมุนจาก IMU (ถ้ามี)
        omega = self.imu_yaw_rate if self.imu_active else self.theta_dot

        # --- Time-Step Integration หาพิกัด (x, y, theta) ---
        x_dot = self.v * math.cos(self.theta)
        y_dot = self.v * math.sin(self.theta)
        
        self.x += x_dot * dt
        self.y += y_dot * dt
        self.theta += omega * dt

        q = self.euler_to_quaternion(0, 0, self.theta)

        # --- TF Broadcaster ---
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

        # --- Publish /custom_odom ---
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]
        odom_msg.twist.twist.linear.x = self.v
        odom_msg.twist.twist.angular.z = omega
        self.odom_pub.publish(odom_msg)

        # --- Publish Path ---
        pose = PoseStamped()
        pose.header.stamp = current_time.to_msg()
        pose.header.frame_id = 'odom'
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.path_msg.poses.append(pose)
        self.path_msg.header.stamp = current_time.to_msg()
        self.path_pub.publish(self.path_msg)

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()