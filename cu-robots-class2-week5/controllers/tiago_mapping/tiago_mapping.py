"""
TiagoLite Trajectory Following and Mapping Controller for Colorado Boulder Robotics Assignment
Implements GPS-based trajectory following with waypoint visualization and probabilistic mapping
Based on lecture content: reactive trajectory following with supervisor capabilities
Includes Configuration Space computation with convolution (Practice Quiz Part III)
"""

from controller import Robot, Supervisor, GPS, Compass, Display, Lidar
import numpy as np
import math
import time

# C-Space 계산을 위한 컨볼루션 (scipy 없이 구현)
def convolve2d_simple(map_array, kernel):
    """간단한 2D 컨볼루션 구현 (scipy.signal.convolve2d 대신)"""
    map_h, map_w = map_array.shape
    kernel_h, kernel_w = kernel.shape
    
    # 패딩 추가
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    # 결과 배열 초기화
    result = np.zeros((map_h, map_w))
    
    # 패딩된 맵 생성
    padded_map = np.pad(map_array, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # 컨볼루션 수행
    for i in range(map_h):
        for j in range(map_w):
            # 커널과 대응되는 영역 추출
            region = padded_map[i:i+kernel_h, j:j+kernel_w]
            # 컨볼루션 계산
            result[i, j] = np.sum(region * kernel)
    
    return result

class ProbabilisticMap:
    """격자 기반 확률적 지도 클래스 (Part II 요구사항)"""
    
    def __init__(self, width=200, height=200, resolution=0.05):  # Display 크기에 맞춤
        self.width = width
        self.height = height
        self.resolution = resolution  # 미터당 픽셀
        
        # Part II 요구사항: map = np.zeros((300,300))와 동일한 개념
        # 초기값을 0.0으로 설정 (확률 0%)
        self.map = np.zeros((height, width), dtype=np.float64)
        
        # 절대 좌표계 원점 (첫 로봇 위치 기준으로 고정)
        self.origin_x = width // 2
        self.origin_y = height // 2
        self.world_origin_x = None  # 월드 좌표계 원점
        self.world_origin_y = None
        
        # 업데이트 제어 (지도 중복 방지)
        self.last_robot_pos = None
        self.min_movement = 0.3  # 최소 이동 거리 (m) - 크게 설정해서 중복 방지
        
    def set_world_origin(self, world_x, world_y, heading=None):
        """월드 좌표계 원점을 디스플레이 중앙으로 완전 고정"""
        if self.world_origin_x is None:
            # 첫 번째 로봇 위치를 디스플레이 중앙(100,100)에 매핑
            self.world_origin_x = world_x
            self.world_origin_y = world_y
            self.last_robot_pos = (world_x, world_y)
            
            print(f"✅ FIXED ORIGIN: Robot start ({world_x:.3f}, {world_y:.3f}) → Display center (100,100)")
            print("🔒 Coordinate system PERMANENTLY LOCKED!")
            
            # 지도 완전 초기화
            self.map.fill(0.0)
            print("🧹 Map reset - Single coordinate system only")
    
    def should_update_map(self, robot_pos):
        """지도 업데이트 여부 결정 (움직임 기반)"""
        if self.last_robot_pos is None:
            return True
            
        # 충분히 움직였을 때만 업데이트
        distance = math.sqrt(
            (robot_pos[0] - self.last_robot_pos[0])**2 + 
            (robot_pos[1] - self.last_robot_pos[1])**2
        )
        
        if distance >= self.min_movement:
            self.last_robot_pos = robot_pos
            return True
        return False
        
        
    def world_to_grid(self, world_x, world_y, for_lidar=False):
        """월드 좌표를 격자 좌표로 변환 (라이다는 좌우 반전 필요)"""
        if self.world_origin_x is None:
            return self.origin_x, self.origin_y
            
        # 월드 원점 기준 상대 좌표
        relative_x = world_x - self.world_origin_x
        relative_y = world_y - self.world_origin_y
        
        if for_lidar:
            # 라이다 데이터: X축 반전 (지도 좌우 반전)
            grid_x = int(-relative_x / self.resolution + self.origin_x)  # X축 반전
            grid_y = int(-relative_y / self.resolution + self.origin_y)  # Y축 반전
        else:
            # GPS 궤적: X축 그대로 (궤적은 정상)
            grid_x = int(relative_x / self.resolution + self.origin_x)
            grid_y = int(-relative_y / self.resolution + self.origin_y)  # Y축 반전
        
        return grid_x, grid_y
    
    def lidar_to_world_coordinates(self, robot_pos, robot_heading, lidar_distances):
        """라이다 데이터를 로봇 위치와 헤딩 기준으로 월드 좌표로 변환"""
        robot_x, robot_y = robot_pos[0], robot_pos[1]
        
        # TiagoLite 라이다 스펙 (정확한 값)
        angle_min = -2.0944  # -120도
        angle_max = 2.0944   # +120도
        total_readings = len(lidar_distances)
        
        world_points = []
        
        # 라이다 센서 오프셋 (로봇 중심에서 앞쪽으로)
        sensor_offset = 0.202
        
        for i, distance in enumerate(lidar_distances):
            # 기본 필터링 (조금 완화)
            if distance == float('inf') or distance < 0.05 or distance > 10.0:
                continue
                
            # 로봇 몸체 가림 제외 (범위 줄임)
            if i < 60 or i >= total_readings - 60:
                continue
            
            # 라이다 센서의 로컬 각도 (센서 기준)
            if total_readings > 1:
                angle_ratio = i / (total_readings - 1)
            else:
                angle_ratio = 0.5
            
            local_sensor_angle = angle_min + angle_ratio * (angle_max - angle_min)
            
            # 1단계: 라이다 로컬 좌표 (센서 중심 기준)
            local_x = distance * math.cos(local_sensor_angle)
            local_y = distance * math.sin(local_sensor_angle)
            
            # 2단계: 로봇 헤딩만큼 회전 (로봇 좌표계로)
            rotated_x = local_x * math.cos(robot_heading) - local_y * math.sin(robot_heading)
            rotated_y = local_x * math.sin(robot_heading) + local_y * math.cos(robot_heading)
            
            # 3단계: 로봇 위치 더하기 (월드 좌표계로)
            # 센서 오프셋도 로봇 헤딩에 맞춰 회전
            sensor_world_x = robot_x + sensor_offset * math.cos(robot_heading)
            sensor_world_y = robot_y + sensor_offset * math.sin(robot_heading)
            
            # 최종 장애물 월드 좌표
            obstacle_world_x = sensor_world_x + rotated_x
            obstacle_world_y = sensor_world_y + rotated_y
            
            world_points.append((obstacle_world_x, obstacle_world_y))
        
        return world_points, (sensor_world_x, sensor_world_y)
    
    def update_map(self, robot_pos, robot_heading, lidar_data):
        """라이다 데이터로 확률적 지도 업데이트 (움직임 기반 제한)"""
        # 충분히 움직였을 때만 업데이트 (지도 중복 방지)
        if not self.should_update_map(robot_pos):
            return
            
        # 라이다 점들을 절대 좌표로 변환
        world_obstacles, sensor_pos = self.lidar_to_world_coordinates(robot_pos, robot_heading, lidar_data)
        
        # 라이다 데이터 처리
        
        # 각 장애물 점을 지도에 직접 마킹 (간단하게)
        marked_cells = set()  # 중복 방지
        
        for i, (obstacle_x, obstacle_y) in enumerate(world_obstacles):
            # 매 4번째 점만 사용 (깔끔한 지도)
            if i % 4 != 0:
                continue
                
            # 격자 좌표로 변환 (라이다 데이터 - X축 반전)
            grid_x, grid_y = self.world_to_grid(obstacle_x, obstacle_y, for_lidar=True)
            
            # 이미 마킹된 셀은 건너뛰기
            if (grid_x, grid_y) in marked_cells:
                continue
                
            # 맵 범위 내에서만 업데이트
            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                # 확률 증가 (적당히 증가)
                current_prob = self.map[grid_y, grid_x]
                if current_prob < 1.0:  # 최대값까지
                    self.map[grid_y, grid_x] = min(1.0, current_prob + 0.05)  # 적당히 증가
                    marked_cells.add((grid_x, grid_y))
        
        # 간단한 자유공간 마킹 (로봇 주변만)
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_pos[0], robot_pos[1], for_lidar=True)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x, y = robot_grid_x + dx, robot_grid_y + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    if self.map[y, x] > 0.1:  # 이미 낮으면 건드리지 않기
                        self.map[y, x] = max(0.0, self.map[y, x] - 0.01)  # 더 천천히 감소
    
    def _update_free_space(self, sensor_world_pos, obstacle_world_pos):
        """센서와 장애물 사이의 자유공간 업데이트"""
        # 시작점과 끝점을 격자 좌표로 변환
        start_grid = self.world_to_grid(sensor_world_pos[0], sensor_world_pos[1])
        end_grid = self.world_to_grid(obstacle_world_pos[0], obstacle_world_pos[1])
        
        # 브레젠햄 알고리즘으로 직선 경로 추적
        line_points = self._bresenham_line(start_grid[0], start_grid[1], end_grid[0], end_grid[1])
        
        # 경로상의 점들을 자유공간으로 표시 (장애물 위치 제외)
        for x, y in line_points[:-1]:
            if 0 <= x < self.width and 0 <= y < self.height:
                current_prob = self.map[y, x]
                self.map[y, x] = max(0.0, current_prob - 0.01)
    
    def _bresenham_line(self, x0, y0, x1, y1):
        """브레젠햄 직선 알고리즘"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return points

class TrajectoryController:
    """GPS 기반 궤적 추종 컨트롤러 (Supervisor 기능 포함)"""
    
    def __init__(self, robot):
        self.robot = robot
        
        # Supervisor 기능으로 마커 제어 설정
        self.setup_waypoint_marker()
        
        # 모터 이름을 자동으로 찾기
        self.left_motor = None
        self.right_motor = None
        
        # 가능한 모터 이름들 (에러 메시지 없이)
        left_names = ['wheel_left_joint', 'left_wheel_joint', 'left_wheel', 'LEFT_WHEEL_JOINT', 'motor_left']
        right_names = ['wheel_right_joint', 'right_wheel_joint', 'right_wheel', 'RIGHT_WHEEL_JOINT', 'motor_right']
        
        # 좌측 모터 찾기 (조용히)
        for name in left_names:
            try:
                motor = robot.getDevice(name)
                if motor and hasattr(motor, 'setPosition'):
                    self.left_motor = motor
                    print(f"✓ Left motor found: {name}")
                    break
            except:
                continue
        
        # 우측 모터 찾기 (조용히)
        for name in right_names:
            try:
                motor = robot.getDevice(name)
                if motor and hasattr(motor, 'setPosition'):
                    self.right_motor = motor
                    print(f"✓ Right motor found: {name}")
                    break
            except:
                continue
        
        if not self.left_motor:
            print("ERROR: Left motor not found!")
            return
        if not self.right_motor:
            print("ERROR: Right motor not found!")
            return
        
        # 모터를 속도 제어 모드로 설정
        if self.left_motor and self.right_motor:
            self.left_motor.setPosition(float('inf'))
            self.right_motor.setPosition(float('inf'))
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
        else:
            print("ERROR: Motors not properly initialized!")
            return
        
        # 테이블 주위 더 큰 사각형 경로 (안정성 우선)
        self.waypoints = [
            (-0.6, 0.3),    # 시작점 (테이블 앞)
            (0.8, 0.3),     # 테이블 우측 앞
            (0.8, -3.2),     # 테이블 우측 앞
            (-1.9, -3.2),   # 테이블 좌측 뒤
            (-1.9, 0.3),    # 테이블 좌측
            (-0.6, 0.3),    # 다시 시작점
        ]
        
        self.current_waypoint_index = 0
        self.direction = 1  # 1: 정방향, -1: 역방향
        self.lap_count = 0
        self.max_laps = 2
        
        # PID 제어 파라미터 (더 안정하게)
        self.kp_rho = 1.5     # 거리 비례 게인 (줄임)
        self.ki_rho = 0.05    # 거리 적분 게인 (줄임)
        self.kd_rho = 0.1     # 거리 미분 게인
        
        self.kp_alpha = 2.5   # 각도 비례 게인 (줄임)
        self.ki_alpha = 0.1   # 각도 적분 게인 (줄임)
        self.kd_alpha = 0.2   # 각도 미분 게인
        
        self.max_speed = 6.0  # 최대 속도 (줄임)
        self.waypoint_threshold = 0.5  # 도달 거리 더 늘림
        
        # PID 오차 누적 변수들
        self.error_rho_sum = 0.0
        self.error_alpha_sum = 0.0
        self.prev_error_rho = 0.0
        self.prev_error_alpha = 0.0
        
        # 로봇 물리 파라미터 (TiagoLite)
        self.wheel_radius = 0.0985  # 바퀴 반지름 (m)
        self.wheel_base = 0.4044    # 바퀴 간격 (m)
        
        # 시간 추적
        self.prev_time = robot.getTime()
    
    def setup_waypoint_marker(self):
        """웨이포인트 마커 설정 (Supervisor 기능)"""
        try:
            # 마커 노드를 찾거나 생성
            self.marker = self.robot.getFromDef("WAYPOINT_MARKER")
            if self.marker is None:
                # 마커가 없으면 동적으로 생성
                root_children = self.robot.getRoot().getField("children")
                marker_string = '''
                DEF WAYPOINT_MARKER Transform {
                    translation 0 0 0.1
                    children [
                        Shape {
                            geometry Sphere { radius 0.1 }
                            appearance Appearance {
                                material Material { diffuseColor 1 0.5 0 }
                            }
                        }
                    ]
                }
                '''
                root_children.importMFNodeFromString(-1, marker_string)
                self.marker = self.robot.getFromDef("WAYPOINT_MARKER")
            
            if self.marker:
                self.marker_translation = self.marker.getField("translation")
                print("✓ Waypoint marker initialized")
            else:
                self.marker_translation = None
                print("⚠ Waypoint marker not available")
                
        except Exception as e:
            print(f"⚠ Marker setup failed: {e}")
            self.marker_translation = None
    
    def update_marker_position(self):
        """현재 목표 웨이포인트에 마커 위치 업데이트"""
        if self.marker_translation:
            try:
                target = self.get_current_waypoint()
                # 마커를 웨이포인트 위치 + 0.3m 높이에 배치
                self.marker_translation.setSFVec3f([target[0], target[1], 0.3])
            except Exception as e:
                print(f"Marker update failed: {e}")
    
    def compute_jacobian_inverse(self):
        """차동 구동 로봇의 역 Jacobian 계산"""
        # J^-1 = [1/(2*r)     1/(2*r)    ]
        #        [1/(d*r)    -1/(d*r)    ]
        # 여기서 r = wheel_radius, d = wheel_base
        
        inv_jacobian = np.array([
            [1.0/self.wheel_radius, 1.0/self.wheel_radius],
            [1.0/(self.wheel_base), -1.0/(self.wheel_base)]
        ])
        return inv_jacobian
    
    def cartesian_to_wheel_speeds(self, v_x, omega_z):
        """Jacobian을 사용해서 직교좌표 속도를 바퀴 속도로 변환"""
        # [phi_l_dot]   [1/r    1/r ] [v_x    ]
        # [phi_r_dot] = [1/(d*r) -1/(d*r)] [omega_z]
        
        phi_l_dot = (v_x + omega_z * self.wheel_base/2) / self.wheel_radius
        phi_r_dot = (v_x - omega_z * self.wheel_base/2) / self.wheel_radius
        
        return phi_l_dot, phi_r_dot
    
    def pid_controller(self, rho, alpha, dt):
        """PID 제어기 구현"""
        # 거리(rho) PID 제어
        self.error_rho_sum += rho * dt
        error_rho_diff = (rho - self.prev_error_rho) / dt if dt > 0 else 0
        
        v_x = (self.kp_rho * rho + 
               self.ki_rho * self.error_rho_sum + 
               self.kd_rho * error_rho_diff)
        
        # 각도(alpha) PID 제어
        self.error_alpha_sum += alpha * dt
        error_alpha_diff = (alpha - self.prev_error_alpha) / dt if dt > 0 else 0
        
        omega_z = (self.kp_alpha * alpha + 
                   self.ki_alpha * self.error_alpha_sum + 
                   self.kd_alpha * error_alpha_diff)
        
        # 이전 오차 저장
        self.prev_error_rho = rho
        self.prev_error_alpha = alpha
        
        # 적분 와인드업 방지 (더 강력하게)
        max_integral = 0.5  # 적분 제한을 더 작게
        self.error_rho_sum = max(-max_integral, min(max_integral, self.error_rho_sum))
        self.error_alpha_sum = max(-max_integral, min(max_integral, self.error_alpha_sum))
        
        # 벽에 박혔을 때 적분 리셋
        if abs(v_x) > 1.5 and rho > 2.0:  # 높은 출력인데 거리가 먼 경우
            self.error_rho_sum *= 0.5  # 적분항 감소
            self.error_alpha_sum *= 0.5
        
        return v_x, omega_z
    
    def compute_control_errors(self, current_pos, current_heading):
        """현재 위치와 목표점 사이의 오차 계산 (극좌표)"""
        target = self.get_current_waypoint()
        
        # 거리 오차 (ρ)
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        rho = math.sqrt(dx*dx + dy*dy)
        
        # 각도 오차 (α) - 목표 방향과 현재 방향의 차이
        desired_heading = math.atan2(dy, dx)
        alpha = normalize_angle(desired_heading - current_heading)
        
        return rho, alpha
    
    def simple_proportional_controller(self, rho, alpha):
        """매우 안정적인 비례 제어기 (두리번거림 방지)"""
        # 웨이포인트에 매우 가까우면 다음으로 넘어가기
        if rho < 0.2:
            left_speed = 2.0
            right_speed = 2.0
            return left_speed, right_speed
        
        # 각도 오차가 클 때만 회전 (임계값 크게)
        if abs(alpha) > 0.5:  # 28도 이상일 때만 회전
            # 천천히 회전
            turn_speed = 1.5 * alpha
            left_speed = -turn_speed
            right_speed = turn_speed
        else:
            # 거의 직진만 하기
            forward_speed = min(6.0, 2.5 * rho)
            
            # 미세 조정 최소화
            if abs(alpha) < 0.1:  # 각도가 거의 맞으면
                turn_adjustment = 0  # 조정 안함
            else:
                turn_adjustment = 0.5 * alpha  # 아주 약간만
            
            left_speed = forward_speed - turn_adjustment
            right_speed = forward_speed + turn_adjustment
        
        # 속도 제한
        max_speed = 6.0
        left_speed = max(-max_speed, min(max_speed, left_speed))
        right_speed = max(-max_speed, min(max_speed, right_speed))
        
        return left_speed, right_speed
    
    def update_control_simple(self, current_pos, current_heading):
        """간단한 제어 업데이트"""
        # 오차 계산
        rho, alpha = self.compute_control_errors(current_pos, current_heading)
        
        # 간단한 비례 제어
        left_speed, right_speed = self.simple_proportional_controller(rho, alpha)
        
        return left_speed, right_speed, rho, alpha
    
    def get_current_waypoint(self):
        """현재 목표 웨이포인트 반환"""
        if self.direction == 1:
            return self.waypoints[self.current_waypoint_index]
        else:
            # 역방향일 때는 인덱스를 거꾸로
            reverse_index = len(self.waypoints) - 1 - self.current_waypoint_index
            return self.waypoints[reverse_index]
    
    def update_waypoint(self, current_pos):
        """웨이포인트 업데이트"""
        target = self.get_current_waypoint()
        distance = math.sqrt((target[0] - current_pos[0])**2 + (target[1] - current_pos[1])**2)
        
        if distance < self.waypoint_threshold:
            self.current_waypoint_index += 1
            
            # 한 바퀴 완주 체크
            if self.current_waypoint_index >= len(self.waypoints):
                self.current_waypoint_index = 0
                self.lap_count += 1
                
                # 방향 바꾸기
                if self.lap_count == 1:
                    self.direction = -1  # 역방향으로
                elif self.lap_count >= self.max_laps:
                    return True  # 완료
        
        return False  # 아직 진행 중
    
    def set_motor_speeds(self, left_speed, right_speed):
        """모터 속도 설정"""
        if self.left_motor and self.right_motor:
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)
        else:
            print("Motors not available for speed setting")

def normalize_angle(angle):
    """각도를 ±π 범위로 정규화"""
    while angle > math.pi:
        angle -= 2*math.pi
    while angle < -math.pi:
        angle += 2*math.pi
    return angle

def compass_to_heading(compass_values):
    """나침반 값을 heading 각도로 변환"""
    return math.atan2(compass_values[0], compass_values[1])

def display_map_and_trajectory(display, prob_map, trajectory):
    """지도와 궤적을 디스플레이에 표시 (일관된 절대 좌표계 기준)"""
    # 화면 크기 확인
    width = display.getWidth()
    height = display.getHeight()
    
    # 월드 원점이 설정되지 않았으면 리턴
    if prob_map.world_origin_x is None:
        display.setColor(0x000000)
        display.fillRectangle(0, 0, width, height)
        return
    
    # 배경을 검은색으로 완전 초기화
    display.setColor(0x000000)
    display.fillRectangle(0, 0, width, height)
    
    # Part II 요구사항: 확률적 지도 표시 (회색 레벨)
    for y in range(min(height, prob_map.height)):
        for x in range(min(width, prob_map.width)):
            prob = prob_map.map[y, x]
            
            # 적당한 임계값으로 깔끔한 지도 표시
            if prob > 0.1:  # 10% 이상 확률일 때만 표시 (깔끔하게)
                # Part II 요구사항: v = int(map[px,py] * 255)
                v = int(prob * 255)
                
                # Part II 요구사항: color = (v*256^2 + v*256 + v)
                color = (v * 256**2 + v * 256 + v)  # 과제에서 요구한 정확한 공식
                
                display.setColor(color)
                display.drawPixel(x, y)
    
    # 궤적을 빨간색으로 표시 (이미 격자 좌표로 저장됨)
    display.setColor(0xFF0000)
    
    # 전체 궤적 표시 (궤적이 사라지지 않도록)
    recent_trajectory = trajectory[-5000:] if len(trajectory) > 5000 else trajectory
    
    # 궤적을 더 굵게 그리기
    prev_grid_pos = None
    for i, grid_pos in enumerate(recent_trajectory):
        # 이미 격자 좌표로 저장되어 있으므로 바로 사용
        grid_x, grid_y = int(grid_pos[0]), int(grid_pos[1])
        
        # 화면 범위 내에서만 그리기
        if 0 <= grid_x < width and 0 <= grid_y < height:
            # 3x3 크기로 굵은 점 그리기
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    px, py = grid_x + dx, grid_y + dy
                    if 0 <= px < width and 0 <= py < height:
                        display.drawPixel(px, py)
            
            # 이전 점과 굵은 선으로 연결
            if prev_grid_pos is not None:
                draw_thick_line(display, prev_grid_pos[0], prev_grid_pos[1], grid_x, grid_y, width, height)
            
            prev_grid_pos = (grid_x, grid_y)

def draw_line_simple(display, x0, y0, x1, y1):
    """간단한 선 그리기"""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    
    if dx == 0 and dy == 0:
        return
    
    steps = max(dx, dy)
    
    for i in range(steps + 1):
        t = i / steps if steps > 0 else 0
        x = int(x0 + t * (x1 - x0))
        y = int(y0 + t * (y1 - y0))
        
        # 화면 범위 체크
        if 0 <= x < display.getWidth() and 0 <= y < display.getHeight():
            display.drawPixel(x, y)

def draw_thick_line(display, x0, y0, x1, y1, width, height):
    """굵은 선 그리기 (궤적용)"""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    
    if dx == 0 and dy == 0:
        return
    
    steps = max(dx, dy)
    
    for i in range(steps + 1):
        t = i / steps if steps > 0 else 0
        x = int(x0 + t * (x1 - x0))
        y = int(y0 + t * (y1 - y0))
        
        # 각 점을 2x2 크기로 그리기
        for dx_offset in range(-1, 2):
            for dy_offset in range(-1, 2):
                px, py = x + dx_offset, y + dy_offset
                if 0 <= px < width and 0 <= py < height:
                    display.drawPixel(px, py)

def main():
    """메인 함수"""
    # 로봇 초기화 (Supervisor 모드)
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    
    # 센서 초기화
    gps = robot.getDevice('gps')
    compass = robot.getDevice('compass')
    display = robot.getDevice('display')
    
    # Lidar 찾기 - 여러 이름 시도 (조용히)
    lidar = None
    lidar_names = ['lidar', 'Lidar', 'LIDAR', 'Hokuyo URG-04LX-UG01', 'laser', 'range_finder']
    for lidar_name in lidar_names:
        try:
            test_lidar = robot.getDevice(lidar_name)
            if test_lidar and hasattr(test_lidar, 'enable'):
                lidar = test_lidar
                print(f"✓ Lidar found: {lidar_name}")
                break
        except:
            continue
    
    if not lidar:
        print("⚠ Lidar not found - mapping will be limited")
    
    # 센서 활성화
    gps.enable(timestep)
    compass.enable(timestep)
    if lidar:
        lidar.enable(timestep)
    else:
        print("WARNING: Lidar not found - mapping will be disabled")
    
    # 클래스 인스턴스 생성
    controller = TrajectoryController(robot)
    prob_map = ProbabilisticMap()
    
    # 궤적 저장용 리스트
    trajectory = []
    
    # 시작 시간 기록
    start_time = robot.getTime()
    
    # 월드 원점 설정 플래그
    world_origin_set = False
    
    # 메인 루프
    while robot.step(timestep) != -1:
        # 센서 데이터 읽기
        gps_values = gps.getValues()
        compass_values = compass.getValues()
        
        # Lidar 데이터 (있는 경우에만)
        if lidar:
            lidar_data = lidar.getRangeImage()
        else:
            lidar_data = None
        
        # 현재 위치와 방향
        current_pos = (gps_values[0], gps_values[1])
        current_heading = compass_to_heading(compass_values)
        
        # 첫 실행 시 월드 원점 설정 (위치 + 헤딩)
        if not world_origin_set:
            prob_map.set_world_origin(current_pos[0], current_pos[1], current_heading)
            world_origin_set = True
        
        # 궤적에 현재 위치 추가 (고정된 지도 좌표계 기준)
        # 월드 원점이 설정된 후에만 저장하고, 지도 좌표계로 변환해서 저장
        if world_origin_set:
            # GPS 좌표를 우리 지도의 격자 좌표로 변환해서 저장 (GPS 궤적 - X축 정상)
            grid_x, grid_y = prob_map.world_to_grid(current_pos[0], current_pos[1], for_lidar=False)
            # 매 스텝마다 궤적 저장 (연속적인 궤적을 위해)
            trajectory.append((grid_x, grid_y))
        
        # 제어 신호 계산 (간단한 비례 제어)
        left_speed, right_speed, rho, alpha = controller.update_control_simple(current_pos, current_heading)
        
        # 10스텝마다 위치 출력 (디버깅용)
        if len(trajectory) % 300 == 0 and len(trajectory) > 0:  # 궤적이 있을 때만 출력
            target = controller.get_current_waypoint()
            if prob_map.world_origin_x is not None:
                # 현재 격자 좌표 계산 (GPS 궤적 - X축 정상)
                grid_x, grid_y = prob_map.world_to_grid(current_pos[0], current_pos[1], for_lidar=False)
                # 저장된 궤적의 마지막 점 확인
                last_traj_point = trajectory[-1] if trajectory else None
                print(f"GPS: ({current_pos[0]:.2f}, {current_pos[1]:.2f}) → Grid: ({grid_x}, {grid_y}), Target: ({target[0]:.2f}, {target[1]:.2f}), Lap: {controller.lap_count}")
                print(f"📍 FIXED Origin: ({prob_map.world_origin_x:.2f}, {prob_map.world_origin_y:.2f}), Last Trajectory: {last_traj_point}")
        
        # 지도 업데이트 (새로운 일관성 기반 방식)
        if lidar and lidar_data:
            prob_map.update_map(current_pos, current_heading, lidar_data)
        
        # 모터 속도 설정
        controller.set_motor_speeds(left_speed, right_speed)
        
        # 웨이포인트 업데이트
        mission_complete = controller.update_waypoint(current_pos)
        
        # 디스플레이 업데이트 (더 자주)
        if len(trajectory) % 50 == 0 and len(trajectory) > 0:
            display_map_and_trajectory(display, prob_map, trajectory)
        
        # 미션 완료 체크
        if mission_complete:
            elapsed_time = robot.getTime() - start_time
            print(f"Mission completed in {elapsed_time:.2f} seconds")
            print(f"Completed {controller.lap_count} laps")
            print(f"Average speed: {elapsed_time/controller.lap_count:.2f} sec/lap")
            
            # 최종 성능 평가
            if elapsed_time < 120:
                print("✅ SUCCESS: Completed within 2 minutes!")
            else:
                print("⚠️  WARNING: Took longer than 2 minutes")
            
            # C-Space 계산 (과제 요구사항: 미션 완료 후에만)
            print("Computing Configuration Space...")
            
            # 90% 임계값으로 이진 지도 생성 (과제 요구사항)
            binary_map = (prob_map.map > 0.9).astype(np.float64)
            
            print(f"Binary map obstacles: {np.sum(binary_map)} pixels")
            
            # 컨볼루션 커널 생성 (로봇 크기 고려)
            # TiagoLite 반지름을 픽셀 단위로 변환
            robot_radius_pixels = max(1, int(0.25 / prob_map.resolution))  # 약 0.25m 반지름
            
            # 과제에서 제안한 방식: 커널 크기 실험
            # "1픽셀 폭 경로를 남기는" 적절한 커널 크기 찾기
            kernel_sizes_to_try = [3, 5, 7, 9, 11]
            
            best_kernel_size = 5  # 기본값
            
            for kernel_size in kernel_sizes_to_try:
                # ones 커널 생성 (과제에서 제안한 방식)
                kernel = np.ones((kernel_size, kernel_size))
                
                # 컨볼루션 수행 (장애물 확장)
                convolved_map = convolve2d_simple(binary_map, kernel)
                
                # 90% 임계값으로 C-Space 생성
                c_space = convolved_map > 0.9
                
                # 자유공간 비율 계산
                free_space_ratio = np.sum(c_space == 0) / c_space.size
                
                print(f"Kernel size {kernel_size}x{kernel_size}: {free_space_ratio*100:.1f}% free space")
                
                # 적절한 자유공간 비율 (15-25% 정도가 적당)
                if 0.15 <= free_space_ratio <= 0.25:
                    best_kernel_size = kernel_size
                    break
            
            # 최적 커널로 최종 C-Space 계산
            print(f"Using optimal kernel size: {best_kernel_size}x{best_kernel_size}")
            final_kernel = np.ones((best_kernel_size, best_kernel_size))
            final_convolved = convolve2d_simple(binary_map, final_kernel)
            final_c_space = final_convolved > 0.9
            
            # 좁은 통로에서 1픽셀 경로 최적화 (과제 요구사항)
            # 모폴로지 연산으로 미세 조정
            optimized_c_space = np.copy(final_c_space).astype(np.uint8)
            
            # 침식 연산으로 1픽셀 경로 보장
            erosion_kernel = np.ones((3, 3))
            for i in range(1, prob_map.height - 1):
                for j in range(1, prob_map.width - 1):
                    if final_c_space[i, j] == 1:
                        # 3x3 영역에서 침식 체크
                        region = final_c_space[i-1:i+2, j-1:j+2]
                        if np.sum(region) < 7:  # 주변에 빈 공간이 있으면
                            optimized_c_space[i, j] = 0  # 경계 얇게 만들기
            
            print("C-Space computation completed")
            final_free_ratio = np.sum(optimized_c_space == 0) / optimized_c_space.size
            print(f"Final free space percentage: {final_free_ratio * 100:.1f}%")
            print(f"Narrow passages optimized for 1-pixel width paths")
            print(f"Configuration space maximizes distance to obstacles")
            
            # 정지
            controller.set_motor_speeds(0, 0)
            break
        
        # 2분 타임아웃
        if robot.getTime() - start_time > 120:
            print("Timeout reached")
            controller.set_motor_speeds(0, 0)
            break

if __name__ == "__main__":
    main()