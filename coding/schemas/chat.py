from strenum import StrEnum
from pydantic import BaseModel


class ChatRole(StrEnum):
    """The role identifying who sent a chat message"""

    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str
