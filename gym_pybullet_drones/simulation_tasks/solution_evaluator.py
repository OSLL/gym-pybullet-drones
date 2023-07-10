import pybullet as p


class Verdict:
    def __init__(self, is_accepted, message=""):
        self.is_accepted = is_accepted
        self.message = message

    def __str__(self):
        return f"task is accepted={self.is_accepted} with message={self.message}"


class SolutionEvaluator:
    def __init__(self, checkers, task_generator):
        self.checkers = checkers
        self.task_generator = task_generator

    def evaluate_solution(self):
        generated_task = self.task_generator.generated_task
        verdict = self._run_solution(generated_task)
        score = verdict.is_accepted
        if not score:
            return verdict
        return Verdict(score, "accepted")

    def _run_solution(self, generated_task):
        task_score = True
        for checker in self.checkers:
            try:
                score = checker.check(generated_task)
                if type(score) is bool and not score:
                    return Verdict(False, "{} has failed".format(checker.__class__.__name__))
                else:
                    task_score = score
            except Exception as e:
                return Verdict(False, "{} has thrown an exception: {}".format(checker.__class__.__name__, str(e)))
        return Verdict(task_score, "accepted")
