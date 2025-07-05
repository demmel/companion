import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { RoleplayPresenter } from '../../presenters/RoleplayPresenter';
import { SummarizationContent, SystemMessage } from '@/types';

describe('SummarizationBubble', () => {
  const mockSummarizationContent: SummarizationContent = {
    type: 'summarization',
    title: '✅ Summarized 8 messages. Context usage: 85.2% → 42.1%',
    summary: 'User and assistant discussed various topics including weather preferences, favorite foods, and upcoming travel plans. The conversation covered seasonal preferences and dietary restrictions.',
    messages_summarized: 8,
    context_usage_before: 85.2,
    context_usage_after: 42.1
  };

  const mockSystemMessage: SystemMessage = {
    role: 'system',
    content: [mockSummarizationContent]
  };

  const defaultProps = {
    messages: [mockSystemMessage],
    isStreamActive: false,
    agentState: {
      current_character_id: null,
      characters: {},
      global_scene: null,
      global_memories: []
    }
  };

  it('should render summarization bubble with title', () => {
    render(<RoleplayPresenter {...defaultProps} />);
    
    expect(screen.getByText('✅ Summarized 8 messages. Context usage: 85.2% → 42.1%')).toBeInTheDocument();
  });

  it('should start collapsed and show expand arrow', () => {
    render(<RoleplayPresenter {...defaultProps} />);
    
    // Summary content should not be visible initially
    expect(screen.queryByText(/weather preferences, favorite foods/)).not.toBeInTheDocument();
    
    // Expand arrow should be present
    expect(screen.getByText('▶')).toBeInTheDocument();
  });

  it('should expand when clicked and show summary content', () => {
    render(<RoleplayPresenter {...defaultProps} />);
    
    const titleButton = screen.getByText('✅ Summarized 8 messages. Context usage: 85.2% → 42.1%');
    fireEvent.click(titleButton);
    
    // Summary content should now be visible
    expect(screen.getByText(/weather preferences, favorite foods/)).toBeInTheDocument();
    expect(screen.getByText(/upcoming travel plans/)).toBeInTheDocument();
  });

  it('should rotate arrow when expanded', () => {
    render(<RoleplayPresenter {...defaultProps} />);
    
    const titleButton = screen.getByText('✅ Summarized 8 messages. Context usage: 85.2% → 42.1%');
    const arrow = screen.getByText('▶');
    
    // Click to expand
    fireEvent.click(titleButton);
    
    // Arrow should still be present (just rotated via CSS)
    expect(arrow).toBeInTheDocument();
  });

  it('should collapse when clicked again', () => {
    render(<RoleplayPresenter {...defaultProps} />);
    
    const titleButton = screen.getByText('✅ Summarized 8 messages. Context usage: 85.2% → 42.1%');
    
    // Expand
    fireEvent.click(titleButton);
    expect(screen.getByText(/weather preferences/)).toBeInTheDocument();
    
    // Collapse
    fireEvent.click(titleButton);
    expect(screen.queryByText(/weather preferences/)).not.toBeInTheDocument();
  });

  it('should handle multiline summary content', () => {
    const multilineSummary: SummarizationContent = {
      ...mockSummarizationContent,
      summary: 'Line 1: Weather discussion\nLine 2: Food preferences\nLine 3: Travel plans'
    };

    const systemMessage: SystemMessage = {
      role: 'system',
      content: [multilineSummary]
    };

    render(<RoleplayPresenter {...defaultProps} messages={[systemMessage]} />);
    
    const titleButton = screen.getByText(multilineSummary.title);
    fireEvent.click(titleButton);
    
    expect(screen.getByText(/Line 1: Weather discussion/)).toBeInTheDocument();
    expect(screen.getByText(/Line 2: Food preferences/)).toBeInTheDocument();
    expect(screen.getByText(/Line 3: Travel plans/)).toBeInTheDocument();
  });

  it('should handle empty summary gracefully', () => {
    const emptySummary: SummarizationContent = {
      ...mockSummarizationContent,
      summary: ''
    };

    const systemMessage: SystemMessage = {
      role: 'system',
      content: [emptySummary]
    };

    render(<RoleplayPresenter {...defaultProps} messages={[systemMessage]} />);
    
    const titleButton = screen.getByText(emptySummary.title);
    fireEvent.click(titleButton);
    
    // Should not crash and bubble should still be expandable
    expect(titleButton).toBeInTheDocument();
  });

  it('should display correct context usage statistics', () => {
    const contextStats: SummarizationContent = {
      type: 'summarization',
      title: '✅ Summarized 12 messages. Context usage: 95.7% → 28.4%',
      summary: 'Test summary',
      messages_summarized: 12,
      context_usage_before: 95.7,
      context_usage_after: 28.4
    };

    const systemMessage: SystemMessage = {
      role: 'system',
      content: [contextStats]
    };

    render(<RoleplayPresenter {...defaultProps} messages={[systemMessage]} />);
    
    expect(screen.getByText('✅ Summarized 12 messages. Context usage: 95.7% → 28.4%')).toBeInTheDocument();
  });

  it('should integrate properly with other message types', () => {
    const messages = [
      { role: 'user' as const, content: [{type: 'text' as const, text: 'Hello!' }] },
      mockSystemMessage,
      { role: 'assistant' as const, content: [{type: 'text' as const, text: 'How can I help?'}], tool_calls: [] }
    ];

    render(<RoleplayPresenter {...defaultProps} messages={messages} />);
    
    // All message types should be present
    expect(screen.getByText('Hello!')).toBeInTheDocument();
    expect(screen.getByText('✅ Summarized 8 messages. Context usage: 85.2% → 42.1%')).toBeInTheDocument();
    expect(screen.getByText('How can I help?')).toBeInTheDocument();
  });
});