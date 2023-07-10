import pybullet as p
import numpy as np
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.enums import DroneModel
from data_history import DataHistory


class DroneEnvironment(CtrlAviary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__is_called_once = False
        self.DEFAULT_CONTROL_FREQ_HZ = 48
        self.drones_history = DataHistory(self)

        # Initialize the controllers use DroneModel from environment
        if self.DRONE_MODEL in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [DSLPIDControl(drone_model=self.DRONE_MODEL) for i in range(self.NUM_DRONES)]
        elif self.DRONE_MODEL in [DroneModel.HB]:
            self.ctrl = [SimplePIDControl(drone_model=self.DRONE_MODEL) for i in range(self.NUM_DRONES)]

    def step(self, target_pos):
        # action is motors rate; dict with pairs {drone number (str) : [4 number for each motors rate]}
        action = {str(i): np.array([0, 0, 0, 0]) for i in range(self.NUM_DRONES)}
        # На самом первом шаге инициализируется состояние системы на 0 шаге
        # в ином случае этого не делаем - сразу получаем информацию об состоянии системы на прошлом шаге
        if not self.__is_called_once:
            observation_vector, _, _, _ = super().step(action)
            self.drones_history.save_drone_data(observation_vector)
            self.__is_called_once = True

        state_vector = self.drones_history.get_states_vector_on_step(-1)

        ctrl_every_n_step = int(np.floor(self.SIM_FREQ / self.DEFAULT_CONTROL_FREQ_HZ))
        # Compute control for the current way point
        for i in range(self.NUM_DRONES):
            action[str(i)], _, _ = self.ctrl[i].computeControlFromState(
                control_timestep=ctrl_every_n_step * self.TIMESTEP,
                state=state_vector[str(i)],
                target_pos=target_pos[i, :]
                )

        # call the step
        observation_vector, _, _, _ = super().step(action)

        # save the drone data (obs vector)
        self.drones_history.save_drone_data(observation_vector)

        super().render()
        return observation_vector

    def get_data_history(self):
        return self.drones_history.get_data_history()
