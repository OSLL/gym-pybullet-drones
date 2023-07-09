
class Checker:
    def __init__(self):
        pass

    def check(self, generated_task, data_to_check):
        raise NotImplementedError


class CrossSquareChecker(Checker):
    def __init__(self, max_deviation=0.1):
        super().__init__()
        self.max_deviation = max_deviation

    def check(self, generated_task) -> bool:
        # square from 0.5 ; 0.5 to 1 ; 1
        data_to_check = generated_task["env"].get_data_history()
        last_position = None
        for obj_data in data_to_check:
            position = obj_data[1]["position"]  # [X;Y;Z]
            last_position = position
            left_corner = 0.5 - self.max_deviation
            right_corner = 1 + self.max_deviation
            if left_corner <= position[0] <= right_corner and left_corner <= position[1] <= right_corner:
                return False
        target = generated_task["target_position"]
        for i in range(3):
            if not (target[i] - self.max_deviation <= last_position[i] <= target[i] + self.max_deviation):
                return False
        return True

