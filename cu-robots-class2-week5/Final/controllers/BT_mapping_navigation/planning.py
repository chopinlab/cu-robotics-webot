import py_trees
import numpy as np
from scipy.spatial.distance import cdist


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
        
        self.logger.debug("  %s [Planning::setup()]" % self.name)
    
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
    
    def astar_path_planning(self, cspace, start_px, goal_px):
        """A* 알고리즘을 사용한 경로 계획"""
        # 간단한 A* 구현
        from heapq import heappush, heappop
        
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        def get_neighbors(pos):
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    x, y = pos[0] + dx, pos[1] + dy
                    if 0 <= x < cspace.shape[0] and 0 <= y < cspace.shape[1]:
                        if cspace[x, y] < 0.9:  # 장애물이 아닌 경우
                            neighbors.append((x, y))
            return neighbors
        
        open_set = []
        heappush(open_set, (0, start_px))
        came_from = {}
        g_score = {start_px: 0}
        f_score = {start_px: heuristic(start_px, goal_px)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == goal_px:
                # 경로 재구성
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_px)
                return path[::-1]
            
            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_px)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # 경로를 찾을 수 없음
    
    def update(self):
        """경로 계획 실행"""
        # 저장된 맵 로드
        try:
            cspace = np.load('cspace.npy')
        except FileNotFoundError:
            print("맵 파일을 찾을 수 없습니다!")
            return py_trees.common.Status.FAILURE
        
        # 현재 위치 획득
        current_pos = self.gps.getValues()
        start_world = (current_pos[0], current_pos[1])
        
        # 좌표 변환
        start_px = tuple(self.world2map(*start_world))
        goal_px = tuple(self.world2map(*self.goal))
        
        # A* 경로 계획 실행
        path_px = self.astar_path_planning(cspace, start_px, goal_px)
        
        if not path_px:
            print("경로를 찾을 수 없습니다!")
            return py_trees.common.Status.FAILURE
        
        # 픽셀 좌표를 월드 좌표로 변환
        path_world = []
        for px, py in path_px[::5]:  # 경로 간소화
            world_coord = self.map2world(px, py)
            path_world.append(tuple(world_coord))
        
        # 목표 지점 추가
        path_world.append(self.goal)
        
        # 블랙보드에 계획된 경로 저장
        self.blackboard.write('waypoints', path_world)
        
        print(f"경로 계획 완료: {len(path_world)}개 웨이포인트")
        return py_trees.common.Status.SUCCESS 