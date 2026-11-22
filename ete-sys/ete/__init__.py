"""ETE - Exam Timetabling Engine: a hybrid CSP / graph-colouring toolkit for exam timetabling."""
from .instance import Instance, Exam, Timetable
from .validate import validate_assignment, count_conflicts

__all__ = ["Instance", "Exam", "Timetable", "validate_assignment", "count_conflicts"]
__version__ = "1.0.0"
