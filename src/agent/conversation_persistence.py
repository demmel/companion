"""
Conversation persistence system with unique IDs and auto-save functionality
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from .types import ConversationData, Message
from .chloe_state import ChloeState


@dataclass
class ConversationMetadata:
    """Metadata for a saved conversation"""
    conversation_id: str
    created_at: str  # ISO format timestamp
    last_updated: str  # ISO format timestamp
    message_count: int
    title: Optional[str] = None  # Optional user-provided title


class ConversationPersistence:
    """Manages conversation persistence with unique IDs"""
    
    def __init__(self, conversations_dir: str = "conversations"):
        self.conversations_dir = Path(conversations_dir)
        self.conversations_dir.mkdir(exist_ok=True)
        
        # Create metadata index file if it doesn't exist
        self.index_file = self.conversations_dir / "index.json"
        if not self.index_file.exists():
            self._save_index({})
    
    def generate_conversation_id(self) -> str:
        """Generate a unique conversation ID based on timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Add microseconds for uniqueness if multiple conversations start at same second
        microseconds = int(time.time() * 1000000) % 1000000
        return f"chloe_{timestamp}_{microseconds:06d}"
    
    def save_conversation(
        self, 
        conversation_id: str, 
        messages: List[Message], 
        chloe_state: ChloeState,
        title: Optional[str] = None
    ) -> None:
        """Save a conversation with its state"""
        conversation_data = ConversationData(messages=messages)
        
        # Save conversation data
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        with open(conversation_file, 'w') as f:
            json.dump(conversation_data.model_dump(), f, indent=2)
        
        # Save Chloe's state separately
        state_file = self.conversations_dir / f"{conversation_id}_state.json"
        with open(state_file, 'w') as f:
            json.dump(chloe_state.model_dump(), f, indent=2)
        
        # Update metadata index
        self._update_metadata(conversation_id, len(messages), title)
    
    def load_conversation(self, conversation_id: str) -> tuple[ConversationData, ChloeState]:
        """Load a conversation and its state"""
        # Load conversation data
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        if not conversation_file.exists():
            raise FileNotFoundError(f"Conversation {conversation_id} not found")
        
        with open(conversation_file, 'r') as f:
            conversation_data = ConversationData.model_validate(json.load(f))
        
        # Load Chloe's state
        state_file = self.conversations_dir / f"{conversation_id}_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                chloe_state = ChloeState.model_validate(json.load(f))
        else:
            # Fallback to default state if no state file exists
            from .chloe_state import create_default_chloe_state
            chloe_state = create_default_chloe_state()
        
        return conversation_data, chloe_state
    
    def list_conversations(self) -> List[ConversationMetadata]:
        """List all saved conversations with metadata"""
        index = self._load_index()
        conversations = []
        
        for conv_id, metadata in index.items():
            conversations.append(ConversationMetadata(
                conversation_id=conv_id,
                created_at=metadata.get('created_at', ''),
                last_updated=metadata.get('last_updated', ''),
                message_count=metadata.get('message_count', 0),
                title=metadata.get('title')
            ))
        
        # Sort by last_updated (most recent first)
        conversations.sort(key=lambda x: x.last_updated, reverse=True)
        return conversations
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its associated files"""
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        state_file = self.conversations_dir / f"{conversation_id}_state.json"
        
        deleted = False
        if conversation_file.exists():
            conversation_file.unlink()
            deleted = True
        
        if state_file.exists():
            state_file.unlink()
        
        # Remove from index
        index = self._load_index()
        if conversation_id in index:
            del index[conversation_id]
            self._save_index(index)
        
        return deleted
    
    def conversation_exists(self, conversation_id: str) -> bool:
        """Check if a conversation exists"""
        conversation_file = self.conversations_dir / f"{conversation_id}.json"
        return conversation_file.exists()
    
    def _update_metadata(self, conversation_id: str, message_count: int, title: Optional[str]) -> None:
        """Update metadata for a conversation"""
        index = self._load_index()
        now = datetime.now().isoformat()
        
        if conversation_id not in index:
            # New conversation
            index[conversation_id] = {
                'created_at': now,
                'last_updated': now,
                'message_count': message_count,
                'title': title
            }
        else:
            # Update existing conversation
            index[conversation_id]['last_updated'] = now
            index[conversation_id]['message_count'] = message_count
            if title is not None:
                index[conversation_id]['title'] = title
        
        self._save_index(index)
    
    def _load_index(self) -> dict:
        """Load the conversation index"""
        if not self.index_file.exists():
            return {}
        
        with open(self.index_file, 'r') as f:
            return json.load(f)
    
    def _save_index(self, index: dict) -> None:
        """Save the conversation index"""
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)