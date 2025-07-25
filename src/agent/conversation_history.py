from typing import List
from agent.types import Message


class ConversationHistory:
    """Interface for managing conversation history during reasoning loop"""

    def __init__(self):
        """Initialize empty conversation history"""
        self.full_history: List[Message] = []
        self.llm_history: List[Message] = []

    def add_message(self, msg: Message):
        """Add message to both full and LLM history appropriately"""
        self.full_history.append(msg)
        self.llm_history.append(msg)

    def get_full_history(self) -> List[Message]:
        """Get full conversation history including all messages"""
        return self.full_history

    def get_summarized_history(self) -> List[Message]:
        """Get history suitable for reasoning context (may be summarized/filtered)"""
        return self.llm_history

    def append(self, msg: Message):
        """Compatibility method for append"""
        self.add_message(msg)

    def insert_summary_notification(
        self, index: int, summary_message: Message, recent_messages: List[Message]
    ):
        """Insert summary notification in full history and replace LLM history for summarization"""
        self.full_history.insert(index, summary_message)
        self.llm_history = [summary_message] + recent_messages

    def __len__(self) -> int:
        """Return length of full history"""
        return len(self.full_history)
