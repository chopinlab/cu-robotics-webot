import py_trees
import numpy as np


class Navigation(py_trees.behaviour.Behaviour):
    """웨이포인트 기반 내비게이션을 수행하는 Behavior 클래스"""
    
    def __init__(self, name, blackboard):
        super(Navigation, self).__init__(name)
        self.blackboard = blackboard
        self.robot = blackboard.read('robot')
    
    def setup(self):
        """내비게이션에 필요한 하드웨어 설정"""
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # GPS 및 나침반 센서 설정
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)
        
        # 모터 설정
        self.leftMotor = self.robot.getDevice('wheel_left_joint')
        self.rightMotor = self.robot.getDevice('wheel_right_joint')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        
        # 마커 설정 (목표 지점 시각화용)
        self.marker = self.robot.getFromDef("marker").getField("translation")
        
        self.logger.debug("  %s [Navigation::setup()]" % self.name)
    
    def initialise(self):
        """내비게이션 초기화"""
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)
        
        self.index = 0
        
        self.logger.debug("  %s [Navigation::initialise()]" % self.name)
        # 블랙보드에서 웨이포인트 읽기
        self.WP = self.blackboard.read('waypoints')
    
    def update(self):
        """내비게이션 업데이트"""
        self.logger.debug("  %s [Navigation::update()]" % self.name)
        
        # 현재 로봇 위치 및 방향 획득
        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]
        theta = np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])
        
        # 현재 목표 웨이포인트까지의 거리 및 각도 계산
        rho = np.sqrt((xw - self.WP[self.index][0])**2 + (yw - self.WP[self.index][1])**2)
        alpha = np.arctan2(self.WP[self.index][1] - yw, self.WP[self.index][0] - xw) - theta
        
        # 각도 정규화
        if alpha > np.pi:
            alpha = alpha - 2*np.pi
        elif alpha < -np.pi:
            alpha = alpha + 2*np.pi
        
        # 목표 마커 위치 설정
        self.marker.setSFVec3f([*self.WP[self.index], 0])
        
        # PID 제어 파라미터
        vL, vR = 6.28, 6.28
        
        p1 = 4  # 각도 제어 게인
        p2 = 2  # 거리 제어 게인
        
        # 모터 속도 계산
        vL = -p1*alpha + p2*rho
        vR = +p1*alpha + p2*rho
        
        # 속도 제한
        vL = min(vL, 6.28)
        vR = min(vR, 6.28)
        vL = max(vL, -6.28)
        vR = max(vR, -6.28)
        
        # 모터에 속도 설정
        self.leftMotor.setVelocity(vL)
        self.rightMotor.setVelocity(vR)
        
        # 웨이포인트 도달 확인
        if rho < 0.4:
            print("Reached ", self.index, len(self.WP))
            self.index = self.index + 1
            if self.index == len(self.WP):
                self.feedback_message = "Last waypoint reached"
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.RUNNING
        else:
            return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        """내비게이션 종료"""
        self.logger.debug("  %s [Navigation::terminate().terminate()][%s->%s]" % 
                         (self.name, self.status, new_status))
        # 로봇 정지
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0) 