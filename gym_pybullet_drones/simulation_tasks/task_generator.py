import pybullet as p
import numpy as np
from gym_pybullet_drones.envs import CtrlAviary
from drone_env import DroneEnvironment
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class TaskGenerator:
    def __init__(self, args):
        self.generated_task = {}
        self.args = args

    def generate_task(self):
        raise NotImplementedError


class SquareTask(TaskGenerator):
    def __init__(self, args):
        super().__init__(args)

    def generate_task(self):
        """ Задача: на сцене расчерчены границы квадрата, Задача дрона - попасть в точку,
            находящуюся по прямой, не попадая в квадрат
            1. Формируем окружение
            2. Заполняем словарь сгенерированных задач
            3. Возвращаем словарь
        """
        INIT_XYZS = np.array([0, 0, 1]).reshape(1,3)
        env = DroneEnvironment(drone_model=DroneModel.CF2X,
                         num_drones=1,
                         initial_xyzs=INIT_XYZS,
                         physics=Physics.PYB,
                         neighbourhood_radius=10,
                         freq=240,
                         aggregate_phy_steps=5,
                         gui=True,
                         record=False,
                         obstacles=False,
                         user_debug_gui=False
                         )
        PYBULLET_CLIENT = env.getPyBulletClient()
        # Создаем очертания квадрата с 0.5;0.5 до 1;1
        p.addUserDebugLine(lineFromXYZ=[0.5, 0.5, 0.1], lineToXYZ=[1, 0.5, 0.1], lineColorRGB=[1, 0, 0], lineWidth=0.3,
                           physicsClientId=PYBULLET_CLIENT)
        p.addUserDebugLine(lineFromXYZ=[1, 0.5, 0.1], lineToXYZ=[1, 1, 0.1], lineColorRGB=[1, 0, 0], lineWidth=0.3,
                           physicsClientId=PYBULLET_CLIENT)
        p.addUserDebugLine(lineFromXYZ=[1, 1, 0.1], lineToXYZ=[0.5, 1, 0.1], lineColorRGB=[1, 0, 0], lineWidth=0.3,
                           physicsClientId=PYBULLET_CLIENT)
        p.addUserDebugLine(lineFromXYZ=[0.5, 1, 0.1], lineToXYZ=[0.5, 0.5, 0.1], lineColorRGB=[1, 0, 0], lineWidth=0.3,
                           physicsClientId=PYBULLET_CLIENT)

        # Добавляем утку в центр
        p.loadURDF("duck_vhacd.urdf", basePosition=[0.75, 0.75, 0.05],
                   baseOrientation=p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=PYBULLET_CLIENT)

        self.generated_task["target_position"] = [1.5, 1.5, 1]
        self.generated_task["env"] = env
        # wrapped_env.render()
        return self.generated_task