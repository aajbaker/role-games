from enum import Enum


class Condition(str, Enum):
    ANONYMOUS = "Anonymous"
    IDENTITY = "Identity"
    TAG_BASED = "Tag-based"
