class Probs():
    def __init__(self,paras):

        if not isinstance(paras._problem_instance, str):
            self.prob = paras._problem_instance
            print("- Prob local loaded ")
        else:
            print("problem "+paras.problem+" not found!")


    def get_problem(self):

        return self.prob
