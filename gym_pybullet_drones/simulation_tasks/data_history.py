import pybullet as p
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from typing import Any


class DroneData:
    """Class for store data about drone. Information consist: position, orientation, euler_orientation,
        velocity, angular_velocity, motors_rate, neighbors, rbg_img, dep_img, seg_img, reward, done, info.
    """

    def __init__(self, obj_id: int, obs: dict, reward: float = None, done: bool = None, info: Any = None):
        """ Parameters
            ----------
            obj_id : int
                Drone id in the PyBullet environment
            obs: dict
                Dictionary with pairs:
                    "state": ndarray (20, ) with position (X;Y;Z), orientation (x;y;z;w),
                    euler_orientation (roll;pitch;yaw), velocity (Vx;Vy;Vz), angular velocity (Wx;Wy;Wz)
                    motors_rate (4 RPMs for each drone motors);
                    "neighbors": list;
                    "rgb", "dep", "seg": image from drone in different format
            reward: float, optional
                Value of the reward function
            done: bool, optional
                Value of the done function
            info: Any, optional
                Additional information about drone
        """
        self.obj_id = obj_id
        self.state = obs["state"]
        self.data = {
            "position": self.state[:3],
            "orientation": self.state[3:7],
            "euler_orientation": self.state[7:10],
            "velocity": self.state[10:13],
            "angular_velocity": self.state[13:16],
            "motors_rate": self.state[16:],
            "neighbors": obs["neighbors"] if "neighbors" in obs.keys() else None,
            "rbg_img": obs["rgb"] if "rgb" in obs.keys() else None,
            "dep_img": obs["dep"] if "dep" in obs.keys() else None,
            "seg_img": obs["seg"] if "seg" in obs.keys() else None,
            "reward": reward,
            "done": done,
            "info": info
        }

    def get_state_vector(self):
        return self.state

    def __getitem__(self, item: str):
        """Provides access to the data dictionary by the attribute key
            Parameters
            ----------
            item: str
                Attribute data to get. Possible value: "position", "orientation","euler_orientation","velocity","angular_velocity",
                "motors_rate", "neighbors", "rbg_img", "dep_img", "seg_img", "reward", "done", "info"

            Returns
            -------
            Any
                Attribute value for drone
        """
        if item in self.data.keys():
            return self.data[item]
        else:
            raise KeyError('No data with key:' + item)

    def __str__(self):
        return f"Drone id: {self.obj_id}\n\
                position (X;Y;Z): {self.data['position']}\n\
                orientation (x;y;z;w): {self.data['orientation']}\n\
                euler orientation (roll;pitch;yaw): {self.data['euler_orientation']}\n\
                velocity (vx;vy;vz): {self.data['velocity']}\n\
                angular velocity (wx;wy;wz): {self.data['angular_velocity']}\n\
                motors rate in RPMs: {self.data['motors_rate']}\n"


class DataHistory:
    """Class for store object data on each step of the simulation """

    def __init__(self, environment):
        """
            Parameters
            ----------
            environment : BaseAviary
                "drone aviary" Gym environment
        """

        """
        data_history: A list with the state of the objects at each step. The i-th index corresponds to the i-th step.
        data_history[i] is the dictionary with pairs of {drone number (int) : object of DroneData class}
        """
        self.data_history = []
        self.__environment = environment

    def save_drone_data(self, obs: dict, reward: dict = None, done: dict = None, info: dict = None) -> None:
        """Method for saving the drone data

            Parameters
            ----------
            obs : Dictionary
                Dictionary with pair {drone_num : Observation vector from step() method of BaseAviary class}
            reward: Dictionary, optional
                Dictionary with pair {drone_num : reward value from step() method of BaseAviary class}
            done: Dictionary, optional
                Dictionary with pair {drone_num : done value from step() method of BaseAviary class}
            info: Dictionary, optional
               Dictionary with pair {drone_num : information from step() method of BaseAviary class}
        """
        drone_ids = self.__environment.getDroneIds()
        data_to_save = dict()
        for drone_id in drone_ids:
            drone_num = drone_id - 1
            drone_obs = obs[str(drone_num)]
            drone_reward = reward[str(drone_num)] if reward and reward[str(drone_num)] else None
            drone_done = done[str(drone_num)] if done and done[str(drone_num)] else None
            drone_info = info[str(drone_num)] if info and info[str(drone_num)] else None
            data_to_save[drone_id] = DroneData(drone_id, drone_obs, drone_reward, drone_done, drone_info)
        self.data_history.append(data_to_save)

    def get_data_history(self) -> list[dict]:
        """Method for get all data history

            Returns
            -------
            list[dict[..]]
                List with data about each drone on each step
        """
        return self.data_history

    def get_states_vector_on_step(self, step) -> dict:
        states_vector = dict()
        number = 0
        for drone_id in self.__environment.getDroneIds():
            states_vector[str(number)] = self.data_history[step][drone_id].get_state_vector()
            number += 1
        return states_vector

    def get_drone_data_on_step(self, step: int, drone_id: int, attribute: str = None) -> Any:
        """Method for get attribute about drone with drone_id on step

        Parameters
        ----------
        step: int
            Step of the simulation
        drone_id: int
            Drone id in PyBullet environment
        attribute: str
            Attribute data to get. Possible value: "position", "orientation","euler_orientation","velocity","angular_velocity",
            "motors_rate", "neighbors", "rbg_img", "dep_img", "seg_img", "reward", "done", "info".
            All possible value also you can see in the DroneData description

        Returns
        -------
        Any
            Attribute value for drone
        """
        if attribute:
            return self.data_history[step][drone_id][attribute]
        else:
            return self.data_history[step][drone_id]
