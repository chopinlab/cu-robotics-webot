from os.path import exists
import py_trees
import numpy as np
from controller import Supervisor
from py_trees.composites import Sequence, Parallel, Selector

# 로컬 모듈 임포트
from navigation import Navigation
from mapping import Mapping
from planning import Planning


class DoesMapExist(py_trees.behaviour.Behaviour):
    """맵 파일이 존재하는지 확인하는 Behavior"""
    
    def update(self):
        file_exists = exists('cspace.npy')
        if file_exists:
            print("맵이 이미 존재합니다")
            return py_trees.common.Status.SUCCESS
        else:
            print("맵이 존재하지 않습니다")
            return py_trees.common.Status.FAILURE


class Blackboard:
    """데이터 교환을 위한 블랙보드 클래스"""
    
    def __init__(self):
        self.data = {}
    
    def write(self, key, value):
        self.data[key] = value
    
    def read(self, key):
        return self.data.get(key)


# 메인 실행부
if __name__ == "__main__":
    # Supervisor 로봇 인스턴스 생성
    robot = Supervisor()
    
    # 시뮬레이션 타임스텝 설정
    timestep = int(robot.getBasicTimeStep())
    
    # 웨이포인트 정의 (테이블 주변 및 목표 지점들)
    WP = [(0.614, -0.19), (0.77, -0.94), (0.37, -3.04),
          (-1.41, -3.39), (-1.53, -3.39), (-1.8, -1.46),
          (-1.44, 0.38), (0, 0)]
    
    # 블랙보드 생성 및 초기 데이터 설정
    blackboard = Blackboard()
    blackboard.write('robot', robot)
    blackboard.write('waypoints', np.concatenate((WP, np.flip(WP, 0)), axis=0))
    
    # Behavior Tree 구성
    tree = Sequence("Main", children=[
        Selector("Does map exist?", children=[
            DoesMapExist("Test for map"),
            Parallel("Mapping", policy=py_trees.common.ParallelPolicy.SuccessOnOne(), children=[
                Mapping("map the environment", blackboard),
                Navigation("move around the table", blackboard)
            ])
        ], memory=True),
        Planning("compute path to lower left corner", blackboard, (-1.46, -3.12)),
        Navigation("move to lower left corner", blackboard),
        Planning("compute path to sink", blackboard, (0.88, 0.09)),
        Navigation("move to sink", blackboard)
    ])
    
    # 메인 시뮬레이션 루프
    tree_setup_with_descendants = tree.setup_with_descendants()
    
    while robot.step(timestep) != -1:
        tree.tick_once()
        
        # 트리가 완료되면 종료
        if tree.status in [py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE]:
            break
    
    print("시뮬레이션 완료!") 