import pybullet as p
import numpy as np
from gym_pybullet_drones.envs import BaseAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.enums import DroneModel
from data_history import DataHistory


class DroneEnvironment:
    def __init__(self, env):
        self.__env = env
        self.__is_called_once = False
        self.drones_history = DataHistory(env)
        self.DEFAULT_CONTROL_FREQ_HZ = 48
        self.NUM_DRONES = self.__env.NUM_DRONES
        self.SIM_FREQ = self.__env.SIM_FREQ
        self.TIMESTEP = 1./self.SIM_FREQ
        self.INIT_XYZS = self.__env.INIT_XYZS

        # Initialize the controllers use DroneModel from environment
        self.num_drones = self.__env.NUM_DRONES
        drone_model = self.__env.DRONE_MODEL
        if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [DSLPIDControl(drone_model=drone_model) for i in range(self.num_drones)]
        elif drone_model in [DroneModel.HB]:
            self.ctrl = [SimplePIDControl(drone_model=drone_model) for i in range(self.num_drones)]

    def step(self, target_pos):
        # action is motors rate; dict with pairs {drone number (str) : [4 number for each motors rate]}
        action = {str(i): np.array([0, 0, 0, 0]) for i in range(self.num_drones)}
        # На самом первом шаге инициализируется состояние системы на 0 шаге
        # в ином случае этого не делаем - сразу получаем информацию об состоянии системы на прошлом шаге
        if not self.__is_called_once:
            observation_vector, _, _, _ = self.__env.step(action)
            self.drones_history.save_drone_data(observation_vector)
            self.__is_called_once = True

        state_vector = self.drones_history.get_states_vector_on_step(-1)

        ctrl_every_n_step = int(np.floor(self.__env.SIM_FREQ / self.DEFAULT_CONTROL_FREQ_HZ))
        # Compute control for the current way point
        for i in range(self.num_drones):
            action[str(i)], _, _ = self.ctrl[i].computeControlFromState(
                control_timestep=ctrl_every_n_step * self.__env.TIMESTEP,
                state=state_vector[str(i)],
                target_pos=target_pos[i, :]
                )

        # call the step
        observation_vector, _, _, _ = self.__env.step(action)

        # save the drone data (obs vector)
        self.drones_history.save_drone_data(observation_vector)

        self.__env.render()
        return observation_vector

    def render(self):
        self.__env.render()

    def close(self):
        self.__env.close()


    def get_data_history(self):
        return self.drones_history.get_data_history()
