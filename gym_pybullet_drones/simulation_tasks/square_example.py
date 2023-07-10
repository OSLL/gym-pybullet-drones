import time

import numpy as np
from task_generator import SquareTask
from gym_pybullet_drones.utils.utils import sync
from solution_evaluator import SolutionEvaluator
from checkers import CrossSquareChecker

class StudentSolution:
    def __init__(self, generated_task):
        self.generated_task = generated_task

    def bad_solve(self):
        env = self.generated_task["env"]
        # Initialize the simulation
        duration_sec = 15
        simulation_freq_hz = env.SIM_FREQ
        control_freq_hz = env.DEFAULT_CONTROL_FREQ_HZ
        AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz)
        CTRL_EVERY_N_STEPS = int(np.floor(simulation_freq_hz / control_freq_hz))
        PERIOD = duration_sec
        NUM_WP = control_freq_hz * PERIOD

        # Initialize the trajectory
        INIT_XYZ = env.INIT_XYZS[0]
        TARGET_XYZ = self.generated_task["target_position"]
        STEP_XYZ = [
            (TARGET_XYZ[0] - INIT_XYZ[0]) / NUM_WP,
            (TARGET_XYZ[1] - INIT_XYZ[1]) / NUM_WP,
            (TARGET_XYZ[2] - INIT_XYZ[2]) / NUM_WP
        ]
        trajectory = np.zeros((NUM_WP, 3))
        trajectory[0, :] = np.array(INIT_XYZ)
        for i in range(1, NUM_WP):
            trajectory[i, :] = trajectory[i-1, :] + STEP_XYZ[:]


        START = time.time()
        wp_counter = 0
        for i in range(0, int(duration_sec*simulation_freq_hz), AGGR_PHY_STEPS):
            if i % CTRL_EVERY_N_STEPS == 0:
                action = np.array([trajectory[wp_counter, :]])
                obs = env.step(action)
                wp_counter = wp_counter + 1 if wp_counter < (NUM_WP - 1) else NUM_WP - 1

            # if i % simulation_freq_hz == 0:
            #     env.render() # see the output info in console

            sync(i, START, env.TIMESTEP)

        env.close()

    def good_solve(self):
        env = self.generated_task["env"]
        # Initialize the simulation
        duration_sec = 15
        simulation_freq_hz = env.SIM_FREQ
        control_freq_hz = env.DEFAULT_CONTROL_FREQ_HZ
        AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz)
        CTRL_EVERY_N_STEPS = int(np.floor(simulation_freq_hz / control_freq_hz))
        PERIOD = duration_sec
        NUM_WP = control_freq_hz * PERIOD

        # Initialize the trajectory
        INIT_XYZ = env.INIT_XYZS[0]
        TARGET_XYZ = self.generated_task["target_position"]
        trajectory = np.zeros((NUM_WP, 3))
        trajectory[0, :] = np.array(INIT_XYZ)
        for i in range(1, NUM_WP):
            if i < 1 * NUM_WP / 3:
                STEP_XYZ = [3 * (TARGET_XYZ[0] - INIT_XYZ[0]) / (1 * NUM_WP), 0, 0]
                trajectory[i, :] = trajectory[i-1, :] + STEP_XYZ[:]
            else:
                STEP_XYZ = [0, 3 * (TARGET_XYZ[1] - INIT_XYZ[1]) / (2 * NUM_WP), 3 * (TARGET_XYZ[2] - INIT_XYZ[2]) / (2 * NUM_WP)]
                trajectory[i, :] = trajectory[i - 1, :] + STEP_XYZ[:]

        START = time.time()
        wp_counter = 0
        for i in range(0, int(duration_sec*simulation_freq_hz), AGGR_PHY_STEPS):
            if i % CTRL_EVERY_N_STEPS == 0:
                action = np.array([trajectory[wp_counter, :]])
                wp_counter = wp_counter + 1 if wp_counter < (NUM_WP - 1) else NUM_WP - 1
                obs = env.step(action)

            # if i % simulation_freq_hz == 0:
            #     env.render() # see the output info in console

            sync(i, START, env.TIMESTEP)

        env.close()


if __name__ == "__main__":
    task_generator = SquareTask(None)
    generated_task = task_generator.generate_task()
    student_solution = StudentSolution(generated_task)
    # student_solution.bad_solve()
    student_solution.good_solve()

    checkers = [CrossSquareChecker()]
    solution_evaluator = SolutionEvaluator(checkers=checkers, task_generator=task_generator)
    verdict = solution_evaluator.evaluate_solution()
    print(verdict)




