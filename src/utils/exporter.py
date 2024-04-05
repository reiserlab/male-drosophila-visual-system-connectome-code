""" Exporter base class """
from abc import ABC, abstractmethod


class Exporter(ABC):
    """
    Abstract base class.

    Currently only used by ExcelExporter, but we might need other exporters in the future.
    """

    @abstractmethod
    def export(self):
        """
        Export. Needs to be implemented in the inheriting class.
        """
