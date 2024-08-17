from abc import abstractmethod
from typing import Dict, Tuple, Union, Optional, List


class StoryBase:
    def __init__(self):
        pass

    @abstractmethod
    def process(self, usr_query: str, history: Optional[List[Dict]] = None):
        raise NotImplementedError
