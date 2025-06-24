import py_trees
import numpy as np


class Planning(py_trees.behaviour.Behaviour):
    """A* 알고리즘을 사용한 경로 계획 Behavior 클래스"""
    
    def __init__(self, name, blackboard, goal):
        super(Planning, self).__init__(name)
        self.blackboard = blackboard
        self.robot = blackboard.read('robot')
        self.goal = goal
    
    def setup(self):
        """경로 계획에 필요한 설정"""
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # GPS 센서 설정
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        
        print(f"  {self.name} [Planning::setup()]")
        return True
    
    def world2map(self, xw, yw):
        """월드 좌표를 맵 픽셀 좌표로 변환"""
        px = int((xw + 2.25) * 40)
        py = int((yw - 2) * (-50))
        
        px = min(px, 199)
        py = min(py, 299)
        px = max(px, 0)
        py = max(py, 0)
        
        return [px, py]
    
    def map2world(self, px, py):
        """맵 픽셀 좌표를 월드 좌표로 변환"""
        xw = px / 40 - 2.25
        yw = py / (-50) + 2
        return [xw, yw]
    
    def simple_path_planning(self, start_world, goal_world):
        """간단한 직선 경로 계획 (A* 대신)"""
        # 중간 웨이포인트들을 생성
        path = []
        num_points = 5
        
        for i in range(num_points + 1):
            t = i / num_points
            x = start_world[0] * (1 - t) + goal_world[0] * t
            y = start_world[1] * (1 - t) + goal_world[1] * t
            path.append((x, y))
        
        return path
    
    def update(self):
        """경로 계획 실행"""
        try:
            # 현재 위치 획득
            current_pos = self.gps.getValues()
            start_world = (current_pos[0], current_pos[1])
            
            print(f"경로 계획: {start_world} -> {self.goal}")
            
            # 간단한 경로 계획 (직선)
            path_world = self.simple_path_planning(start_world, self.goal)
            
            # 블랙보드에 계획된 경로 저장
            self.blackboard.write('waypoints', path_world)
            
            print(f"경로 계획 완료: {len(path_world)}개 웨이포인트")
            return py_trees.common.Status.SUCCESS
            
        except Exception as e:
            print(f"경로 계획 중 에러: {e}")
            return py_trees.common.Status.FAILURE 