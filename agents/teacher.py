from abc import ABC, abstractmethod
from typing import Any

class Teacher(ABC):
	"""
	Act instead of the agent
	"""
	@abstractmethod
	def act(self, observation: Any, info: dict) -> Any:
		pass

	"""
	Decide if the teacher has to act
	"""
	@abstractmethod
	def should_act(self, observation: Any, info: dict) -> Any:
		return False