import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

class KinematicsCalculator(Node):
    def __init__(self):
        super().__init__('kinematics_node')
        
        # 1. Subscribe ค่าจาก /joint_states
        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
            
        # 2. Publisher สำหรับส่งค่า v และ w ออกไป
        self.publisher_ = self.create_publisher(Twist, '/calculated_vel', 10)

        # ตัวแปรเก็บค่ารอบที่แล้ว (prev)
        self.prev_time = None
        self.prev_pos_L = 0.0
        self.prev_pos_R = 0.0

        # +++ เพิ่มตัวแปรเก็บ "ระยะทางรวม" เริ่มต้นที่ 0 เมตร +++
        self.total_distance = 0.0 

        # ค่า Constants 
        self.L = 0.160 # Wheel Base
        self.r = 0.033 # รัศมีล้อ TurtleBot3 Burger (เมตร)

    def joint_callback(self, msg):
        # ดึงเวลาปัจจุบัน (แปลงเป็นวินาที)
        current_time = msg.header.stamp.sec + (msg.header.stamp.nanosec / 1e9)
        
        # ดึงค่าเรเดียนของล้อซ้ายและขวา
        curr_pos_L = msg.position[0]
        curr_pos_R = msg.position[1]

        # ถ้าเป็นการรันรอบแรก ให้จำค่าเริ่มต้นไว้ก่อน แล้วข้ามไปก่อน
        if self.prev_time is None:
            self.prev_time = current_time
            self.prev_pos_L = curr_pos_L
            self.prev_pos_R = curr_pos_R
            return

        # --- เริ่มขั้นตอนการคำนวณ ---
        
        # หา dt
        dt = current_time - self.prev_time
        if dt <= 0: return

        # Step 2: หาระยะทางขยับของล้อแต่ละข้าง (ds)
        ds_L = self.r * (curr_pos_L - self.prev_pos_L)
        ds_R = self.r * (curr_pos_R - self.prev_pos_R)

        # Step 3: คำนวณ v และ w
        v = (ds_R + ds_L) / 2.0 / dt
        w = (ds_R - ds_L) / self.L / dt

        # +++ คำนวณระยะทางที่ขยับได้ในลูปนี้ แล้วนำไปบวกสะสม +++
        # ใช้ abs() เพื่อให้ค่าเป็นบวกเสมอ (เหมือนเลขไมล์รถ ถอยหลังเลขก็ยังเพิ่ม)
        d_center = abs((ds_R + ds_L) / 2.0) 
        self.total_distance += d_center

        # --- เตรียม Publish ค่า ---
        vel_msg = Twist()
        vel_msg.linear.x = float(v)   
        vel_msg.angular.z = float(w)  
        
        self.publisher_.publish(vel_msg)
        
        # +++ Print ระยะทางรวมโชว์ใน Terminal ด้วย +++
        self.get_logger().info(f'v: {v:.3f} m/s | w: {w:.3f} rad/s | Dist: {self.total_distance:.3f} m')

        # อัปเดตค่า prev สำหรับลูปรอบถัดไป
        self.prev_time = current_time
        self.prev_pos_L = curr_pos_L
        self.prev_pos_R = curr_pos_R

def main(args=None):
    rclpy.init(args=args)
    node = KinematicsCalculator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()