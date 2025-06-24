from os.path import exists
import numpy as np
from controller import Supervisor

# 로컬 모듈 임포트
from navigation import Navigation
from mapping import Mapping
from planning import Planning


# 간단한 BT 상태 정의
class Status:
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class DoesMapExist:
    """맵 파일이 존재하는지 확인하는 Behavior"""
    
    def __init__(self, name):
        self.name = name
        self.status = Status.RUNNING
    
    def setup(self):
        return True
    
    def update(self):
        file_exists = exists('cspace.npy')
        if file_exists:
            print("맵이 이미 존재합니다")
            self.status = Status.SUCCESS
            return Status.SUCCESS
        else:
            print("맵이 존재하지 않습니다")
            self.status = Status.FAILURE
            return Status.FAILURE


class Selector:
    """선택자 노드 - 자식이 성공할 때까지 실행"""
    
    def __init__(self, name, children):
        self.name = name
        self.children = children
        self.current_child = 0
        self.status = Status.RUNNING
    
    def setup(self):
        for child in self.children:
            if hasattr(child, 'setup'):
                child.setup()
        return True
    
    def update(self):
        for i in range(self.current_child, len(self.children)):
            child_status = self.children[i].update()
            
            if child_status == Status.SUCCESS:
                self.status = Status.SUCCESS
                self.current_child = 0
                return Status.SUCCESS
            elif child_status == Status.RUNNING:
                self.current_child = i
                self.status = Status.RUNNING
                return Status.RUNNING
            # FAILURE면 다음 자식으로
        
        # 모든 자식이 실패
        self.status = Status.FAILURE
        self.current_child = 0
        return Status.FAILURE


class Sequence:
    """시퀀스 노드 - 모든 자식이 성공해야 성공"""
    
    def __init__(self, name, children):
        self.name = name
        self.children = children
        self.current_child = 0
        self.status = Status.RUNNING
    
    def setup(self):
        for child in self.children:
            if hasattr(child, 'setup'):
                child.setup()
        return True
    
    def update(self):
        while self.current_child < len(self.children):
            child_status = self.children[self.current_child].update()
            
            if child_status == Status.SUCCESS:
                self.current_child += 1
                if self.current_child >= len(self.children):
                    self.status = Status.SUCCESS
                    return Status.SUCCESS
            elif child_status == Status.RUNNING:
                self.status = Status.RUNNING
                return Status.RUNNING
            else:  # FAILURE
                self.status = Status.FAILURE
                self.current_child = 0
                return Status.FAILURE
        
        self.status = Status.SUCCESS
        return Status.SUCCESS


class Parallel:
    """병렬 노드 - SuccessOnOne 정책"""
    
    def __init__(self, name, children):
        self.name = name
        self.children = children
        self.status = Status.RUNNING
        self.success_found = False
    
    def setup(self):
        for child in self.children:
            if hasattr(child, 'setup'):
                child.setup()
        return True
    
    def update(self):
        if self.success_found:
            return Status.SUCCESS
            
        all_finished = True
        
        for child in self.children:
            child_status = child.update()
            
            if child_status == Status.SUCCESS:
                self.success_found = True
                # 다른 자식들 종료
                for other_child in self.children:
                    if hasattr(other_child, 'terminate') and other_child != child:
                        other_child.terminate(Status.SUCCESS)
                self.status = Status.SUCCESS
                return Status.SUCCESS
            elif child_status == Status.RUNNING:
                all_finished = False
        
        if all_finished and not self.success_found:
            self.status = Status.FAILURE
            return Status.FAILURE
        else:
            self.status = Status.RUNNING
            return Status.RUNNING


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
    tree = Sequence("Main", [
        Selector("Does map exist?", [
            DoesMapExist("Test for map"),
            Parallel("Mapping", [
                Mapping("map the environment", blackboard),
                Navigation("move around the table", blackboard)
            ])
        ]),
        Planning("compute path to lower left corner", blackboard, (-1.46, -3.12)),
        Navigation("move to lower left corner", blackboard),
        Planning("compute path to sink", blackboard, (0.88, 0.09)),
        Navigation("move to sink", blackboard)
    ])
    
    # 메인 시뮬레이션 루프
    try:
        tree.setup()
        print("Behavior Tree 초기화 완료")
        
        while robot.step(timestep) != -1:
            status = tree.update()
            
            # 트리가 완료되면 종료
            if status in [Status.SUCCESS, Status.FAILURE]:
                print(f"트리 실행 완료: {status}")
                break
                
    except Exception as e:
        print(f"실행 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("시뮬레이션 완료!") 