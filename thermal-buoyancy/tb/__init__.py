import abc


class AbstractFormulation(abc.ABC):

    @abc.abstractmethod
    def ufl_element(self):
        pass

    @property
    @abc.abstractmethod
    def function_space(self):
        pass

    @abc.abstractmethod
    def create_soln_var(self):
        pass

    @abc.abstractmethod
    def formulate(self, mu, f, soln_var):
        pass

    @abc.abstractmethod
    def create_bcs(self):
        pass
