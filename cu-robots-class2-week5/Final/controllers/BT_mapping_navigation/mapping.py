import py_trees
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def world2map(xw, yw):
    """월드 좌표를 맵 픽셀 좌표로 변환"""
    px = int((xw + 2.25) * 40)
    py = int((yw - 2) * (-50))
    
    # 경계값 제한
    px = min(px, 199)
    py = min(py, 299)
    px = max(px, 0)
    py = max(py, 0)
    
    return [px, py]


def map2world(px, py):
    """맵 픽셀 좌표를 월드 좌표로 변환"""
    xw = px / 40 - 2.25
    yw = py / (-50) + 2
    return [xw, yw]


class Mapping(py_trees.behaviour.Behaviour):
    """환경 매핑을 수행하는 Behavior 클래스"""
    
    def __init__(self, name, blackboard):
        super(Mapping, self).__init__(name)
        self.blackboard = blackboard
        self.hasrun = False
        self.robot = blackboard.read('robot')
    
    def setup(self):
        """매핑에 필요한 센서들 초기화"""
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # GPS 센서 설정
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        
        # 나침반 센서 설정
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)
        
        # 라이다 센서 설정
        self.lidar = self.robot.getDevice('Hokuyo URG-04LX-UG01')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        
        # 디스플레이 설정
        self.display = self.robot.getDevice('display')
        
        self.logger.debug("  %s [Mapping::setup()]" % self.name)
        
        # 맵 초기화 (200x300 픽셀)
        self.map = np.zeros((200, 300))
        
        # 라이다 각도 배열 설정
        self.angles = np.linspace(4.19/2, -4.19/2, 667)
        self.angles = self.angles[80:len(self.angles)-80]
    
    def update(self):
        """매핑 업데이트 수행"""
        self.hasrun = True
        
        # 로봇 현재 위치 및 방향 획득
        xw = self.gps.getValues()[0]
        yw = self.gps.getValues()[1]
        theta = np.arctan2(self.compass.getValues()[0], self.compass.getValues()[1])
        
        # 로봇 위치를 맵에 표시
        px, py = world2map(xw, yw)
        self.display.setColor(0xFF0000)  # 빨간색
        self.display.drawPixel(px, py)
        
        # 변환 행렬 생성 (로봇 좌표계에서 월드 좌표계로)
        w_T_r = np.array([[np.cos(theta), -np.sin(theta), xw],
                          [np.sin(theta), np.cos(theta), yw],
                          [0, 0, 1]])
        
        # 라이다 데이터 획득 및 전처리
        ranges = np.array(self.lidar.getRangeImage())
        ranges = ranges[80:len(ranges)-80]
        ranges[ranges == np.inf] = 100
        
        # 라이다 포인트를 로봇 좌표계에서 계산
        X_i = np.array([ranges * np.cos(self.angles) + 0.202, 
                        ranges * np.sin(self.angles), 
                        np.ones(len(ranges))])
        
        # 월드 좌표계로 변환
        D = w_T_r @ X_i
        
        # 장애물 점들을 맵에 추가
        for d in D.transpose():
            px, py = world2map(d[0], d[1])
            self.map[px, py] += 0.01
            if self.map[px, py] > 1:
                self.map[px, py] = 1
            
            # 점유 확률에 따른 색상으로 디스플레이에 표시
            v = int(self.map[px, py] * 255)
            color = (v * 256**2 + v * 256 + v)
            self.display.setColor(int(color))
            self.display.drawPixel(px, py)
        
        return py_trees.common.Status.RUNNING
    
    def terminate(self, new_status):
        """매핑 종료 시 맵을 저장"""
        self.logger.debug("  %s [Foo::terminate().terminate()][%s->%s]" % 
                         (self.name, self.status, new_status))
        if self.hasrun:
            # Configuration Space 생성 및 저장
            cspace = signal.convolve2d(self.map, np.ones((26, 26)), mode='same')
            plt.figure(0)
            plt.imshow(cspace)
            plt.show()
            
            # 맵을 디스크에 저장
            np.save('cspace', cspace)
            print("맵이 저장되었습니다!") 