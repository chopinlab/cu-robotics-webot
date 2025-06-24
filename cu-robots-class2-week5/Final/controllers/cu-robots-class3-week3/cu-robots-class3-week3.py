"""
로봇 내비게이션 및 매핑 시뮬레이션
Author: Modified Version
"""

import numpy as np
import matplotlib.pyplot as plt
from controller import Supervisor
from scipy import signal


class RobotNavigationSystem:
    """로봇 내비게이션 및 매핑을 위한 클래스"""
    
    def __init__(self):
        # 시스템 파라미터 설정
        self.setup_parameters()
        
        # 하드웨어 초기화
        self.initialize_hardware()
        
        # 매핑 관련 초기화
        self.initialize_mapping()
        
        # 내비게이션 상태 초기화
        self.reset_navigation_state()
    
    def setup_parameters(self):
        """시스템 파라미터 설정"""
        # 로봇 운동 파라미터
        self.max_speed = 6.28
        self.angular_control_gain = 5
        self.linear_control_gain = 3
        self.proximity_threshold = 0.32
        
        # 월드 좌표계 설정
        self.world_dimensions = {
            'width': 4.0,
            'height': 6.0
        }
        
        # 로봇 초기 위치
        self.initial_pose = {
            'x': -0.300,
            'y': 0.425,
            'theta': 0.0
        }
        
        # 경로점 정의
        self.waypoint_list = [
            (0.461, -0.119), 
            (0.80, -0.92), 
            (0.34, -3.13), 
            (0.00, -3.2),
            (-1.33, -3.39), 
            (-1.51, -3.42), 
            (-1.71, -1.47),
            (-1.41, 0.38), 
            (0, 0)
        ]
    
    def initialize_hardware(self):
        """하드웨어 장치 초기화"""
        # 수퍼바이저 생성
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        # 모터 설정
        self.setup_motors()
        
        # 센서 설정
        self.setup_sensors()
        
        # 노드 참조 획득
        self.setup_node_references()
        
        # 라이다 구성
        self.configure_lidar()
    
    def setup_motors(self):
        """모터 장치 설정"""
        self.left_wheel = self.supervisor.getDevice('wheel_left_joint')
        self.right_wheel = self.supervisor.getDevice('wheel_right_joint')
        
        # 무한 위치 제어 모드로 설정
        self.left_wheel.setPosition(float('inf'))
        self.right_wheel.setPosition(float('inf'))
    
    def setup_sensors(self):
        """센서 장치 설정"""
        # 라이다 센서
        self.lidar = self.supervisor.getDevice('Hokuyo URG-04LX-UG01')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        
        # 위치 센서
        self.gps = self.supervisor.getDevice('gps')
        self.gps.enable(self.timestep)
        
        # 방향 센서
        self.compass = self.supervisor.getDevice('compass')
        self.compass.enable(self.timestep)
        
        # 디스플레이
        self.display = self.supervisor.getDevice('display')
    
    def setup_node_references(self):
        """씬 노드 참조 설정"""
        # 마커 노드
        self.marker_node = self.supervisor.getFromDef("marker")
        self.marker_position = self.marker_node.getField('translation')
        self.marker_rotation = self.marker_node.getField('rotation')
        
        # 로봇 노드
        self.robot_node = self.supervisor.getFromDef("tiago_lite")
        self.robot_position = self.robot_node.getField('translation')
        self.robot_rotation = self.robot_node.getField('rotation')
    
    def configure_lidar(self):
        """라이다 구성 설정"""
        # 라이다 스펙
        self.lidar_specs = {
            'fov_degrees': 240.0,
            'step_degrees': 0.36,
            'trim_count': 80
        }
        
        # 해상도 계산
        total_resolution = round(self.lidar_specs['fov_degrees'] / self.lidar_specs['step_degrees'])
        self.active_resolution = total_resolution - 2 * self.lidar_specs['trim_count']
        active_fov = self.active_resolution * self.lidar_specs['step_degrees']
        
        # 라이다 오프셋
        lidar_node = self.supervisor.getFromDef("LIDAR")
        lidar_translation = lidar_node.getField('translation').getSFVec3f()
        self.lidar_offset = {
            'x': lidar_translation[0],
            'y': lidar_translation[1]
        }
        
        # 스캔 각도 배열
        self.scan_angles = np.linspace(
            np.radians(active_fov / 2),
            np.radians(-active_fov / 2),
            self.active_resolution
        )
    
    def initialize_mapping(self):
        """매핑 시스템 초기화"""
        display_width = self.display.getWidth()
        display_height = self.display.getHeight()
        
        self.occupancy_map = np.zeros((display_width, display_height))
        
        # 컨볼루션 커널 크기 결정
        kernel_size = 30 if (display_width, display_height) == (200, 300) else 60
        self.convolution_kernel = np.ones((kernel_size, kernel_size))
    
    def reset_navigation_state(self):
        """내비게이션 상태 초기화"""
        self.current_waypoint_idx = 0
        self.navigation_cycle = 0
        
        # 로봇 초기 위치 설정
        self.robot_position.setSFVec3f([
            self.initial_pose['x'],
            self.initial_pose['y'],
            0.15
        ])
        self.robot_rotation.setSFRotation([0, 0, 1, self.initial_pose['theta']])
    
    @staticmethod
    def convert_world_to_pixel(world_x, world_y, map_shape, world_dims):
        """월드 좌표를 픽셀 좌표로 변환"""
        map_w, map_h = map_shape
        world_w, world_h = world_dims
        
        # 좌표 변환
        pixel_x = int((world_x + 2.5) * map_w / world_w)
        pixel_y = int(-(world_y - 2) * map_h / world_h)
        
        # 경계 제한
        pixel_x = np.clip(pixel_x, 0, map_w - 1)
        pixel_y = np.clip(pixel_y, 0, map_h - 1)
        
        return pixel_x, pixel_y
    
    @staticmethod
    def wrap_angle(angle):
        """각도를 [-π, π] 범위로 정규화"""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def get_robot_pose(self):
        """현재 로봇 위치 및 방향 획득"""
        position = self.gps.getValues()
        compass_values = self.compass.getValues()
        
        return {
            'x': position[0],
            'y': position[1],
            'theta': np.arctan2(compass_values[0], compass_values[1])
        }
    
    def update_target_marker(self):
        """타겟 마커 위치 업데이트"""
        if self.current_waypoint_idx < len(self.waypoint_list):
            target_waypoint = self.waypoint_list[self.current_waypoint_idx]
            self.marker_position.setSFVec3f([*target_waypoint, 0])
            self.marker_rotation.setSFRotation([0, 0, 1, 0])
    
    def calculate_navigation_command(self, robot_pose):
        """내비게이션 명령 계산"""
        if self.current_waypoint_idx >= len(self.waypoint_list):
            return 0, 0, float('inf')
        
        target = self.waypoint_list[self.current_waypoint_idx]
        
        # 거리 및 각도 계산
        dx = target[0] - robot_pose['x']
        dy = target[1] - robot_pose['y']
        distance = np.sqrt(dx**2 + dy**2)
        
        angle_to_target = np.arctan2(dy, dx) - robot_pose['theta']
        angle_to_target = self.wrap_angle(angle_to_target)
        
        return distance, angle_to_target, distance
    
    def update_waypoint_progress(self, distance_to_target):
        """웨이포인트 진행 상태 업데이트"""
        if distance_to_target < self.proximity_threshold:
            if self.navigation_cycle == 0:
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx >= len(self.waypoint_list):
                    self.navigation_cycle += 1
            elif self.navigation_cycle == 1:
                self.current_waypoint_idx -= 1
                if self.current_waypoint_idx < 0:
                    self.navigation_cycle += 1
    
    def get_lidar_data(self):
        """라이다 데이터 획득 및 전처리"""
        raw_ranges = np.array(self.lidar.getRangeImage())
        
        # 트림 및 무한값 처리
        trimmed_ranges = raw_ranges[self.lidar_specs['trim_count']:-self.lidar_specs['trim_count']]
        trimmed_ranges[trimmed_ranges == np.inf] = 100
        
        return trimmed_ranges
    
    def transform_lidar_to_world(self, ranges, robot_pose):
        """라이다 데이터를 월드 좌표계로 변환"""
        # 로봇 프레임에서 라이다 포인트 계산
        local_points = np.array([
            ranges * np.cos(self.scan_angles) + self.lidar_offset['x'],
            ranges * np.sin(self.scan_angles) + self.lidar_offset['y'],
            np.ones(self.active_resolution)
        ])
        
        # 변환 행렬 생성
        cos_theta = np.cos(robot_pose['theta'])
        sin_theta = np.sin(robot_pose['theta'])
        
        transform_matrix = np.array([
            [cos_theta, -sin_theta, robot_pose['x']],
            [sin_theta,  cos_theta, robot_pose['y']],
            [0, 0, 1]
        ])
        
        # 월드 좌표계로 변환
        world_points = transform_matrix @ local_points
        
        return world_points
    
    def update_occupancy_map(self, world_points, robot_pose):
        """점유 지도 업데이트"""
        # 로봇 위치 표시
        robot_px, robot_py = self.convert_world_to_pixel(
            robot_pose['x'], robot_pose['y'],
            self.occupancy_map.shape,
            (self.world_dimensions['width'], self.world_dimensions['height'])
        )
        
        self.display.setColor(0xFF0000)  # 빨간색
        self.display.drawPixel(robot_px, robot_py)
        
        # 장애물 점들 처리
        for point in world_points.T:
            obs_px, obs_py = self.convert_world_to_pixel(
                point[0], point[1],
                self.occupancy_map.shape,
                (self.world_dimensions['width'], self.world_dimensions['height'])
            )
            
            # 점유 확률 업데이트
            self.occupancy_map[obs_px, obs_py] += 0.01
            self.occupancy_map[obs_px, obs_py] = min(1.0, self.occupancy_map[obs_px, obs_py])
            
            # 디스플레이에 표시
            intensity = int(self.occupancy_map[obs_px, obs_py] * 255)
            color = intensity * (256**2 + 256 + 1)
            self.display.setColor(int(color))
            self.display.drawPixel(obs_px, obs_py)
    
    def compute_motor_speeds(self, distance, angle):
        """모터 속도 계산"""
        left_speed = -angle * self.angular_control_gain + distance * self.linear_control_gain
        right_speed = angle * self.angular_control_gain + distance * self.linear_control_gain
        
        # 속도 제한
        left_speed = np.clip(left_speed, -self.max_speed, self.max_speed)
        right_speed = np.clip(right_speed, -self.max_speed, self.max_speed)
        
        return left_speed, right_speed
    
    def set_motor_velocities(self, left_speed, right_speed):
        """모터 속도 설정"""
        self.left_wheel.setVelocity(left_speed)
        self.right_wheel.setVelocity(right_speed)
    
    def stop_robot(self):
        """로봇 정지"""
        self.set_motor_velocities(0, 0)
        print("로봇이 최종 목적지에 도착했습니다!")
    
    def is_mission_complete(self, distance_to_target):
        """미션 완료 여부 확인"""
        return (self.navigation_cycle >= 2 and 
                distance_to_target < self.proximity_threshold)
    
    def generate_final_map(self):
        """최종 지도 생성 및 표시"""
        convolved_map = signal.convolve2d(
            self.occupancy_map, 
            self.convolution_kernel, 
            mode='same'
        )
        
        plt.imshow(convolved_map > 0.9, origin='lower')
        plt.title("점유 지도 (C-Space 적용)")
        plt.xlabel("맵 X 좌표 (픽셀)")
        plt.ylabel("맵 Y 좌표 (픽셀)")
        plt.show()
    
    def run_simulation(self):
        """메인 시뮬레이션 실행"""
        while self.supervisor.step(self.timestep) != -1:
            # 현재 로봇 상태 획득
            current_pose = self.get_robot_pose()
            
            # 타겟 마커 업데이트
            self.update_target_marker()
            
            # 내비게이션 명령 계산
            distance_to_target, angle_to_target, _ = self.calculate_navigation_command(current_pose)
            
            # 웨이포인트 진행 상태 업데이트
            self.update_waypoint_progress(distance_to_target)
            
            # 라이다 데이터 처리
            lidar_ranges = self.get_lidar_data()
            world_points = self.transform_lidar_to_world(lidar_ranges, current_pose)
            
            # 지도 업데이트
            self.update_occupancy_map(world_points, current_pose)
            
            # 미션 완료 확인
            if self.is_mission_complete(distance_to_target):
                self.stop_robot()
                break
            
            # 모터 제어
            left_speed, right_speed = self.compute_motor_speeds(distance_to_target, angle_to_target)
            self.set_motor_velocities(left_speed, right_speed)
        
        # 최종 지도 생성
        self.generate_final_map()


# 프로그램 실행
if __name__ == "__main__":
    navigation_system = RobotNavigationSystem()
    navigation_system.run_simulation()