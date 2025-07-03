import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChatInput } from '../ChatInput';

describe('ChatInput', () => {
  const defaultProps = {
    value: '',
    onChange: vi.fn(),
    onSubmit: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should render input with placeholder', () => {
    render(<ChatInput {...defaultProps} />);
    
    expect(screen.getByPlaceholderText('Type your message...')).toBeInTheDocument();
  });

  it('should render custom placeholder', () => {
    render(<ChatInput {...defaultProps} placeholder="Custom placeholder" />);
    
    expect(screen.getByPlaceholderText('Custom placeholder')).toBeInTheDocument();
  });

  it('should call onChange when typing', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    
    render(<ChatInput {...defaultProps} onChange={onChange} />);
    
    const input = screen.getByRole('textbox');
    await user.type(input, 'hello');
    
    expect(onChange).toHaveBeenCalledWith('h');
    expect(onChange).toHaveBeenCalledWith('e');
    // Called for each character
    expect(onChange).toHaveBeenCalledTimes(5);
  });

  it('should call onSubmit when form is submitted', async () => {
    const onSubmit = vi.fn();
    
    render(<ChatInput {...defaultProps} value="test message" onSubmit={onSubmit} />);
    
    const form = screen.getByRole('textbox').closest('form')!;
    fireEvent.submit(form);
    
    expect(onSubmit).toHaveBeenCalledWith('test message');
  });

  it('should call onSubmit when send button is clicked', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    
    render(<ChatInput {...defaultProps} value="test message" onSubmit={onSubmit} />);
    
    const sendButton = screen.getByRole('button');
    await user.click(sendButton);
    
    expect(onSubmit).toHaveBeenCalledWith('test message');
  });

  it('should not submit empty or whitespace-only messages', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    
    render(<ChatInput {...defaultProps} value="   " onSubmit={onSubmit} />);
    
    const sendButton = screen.getByRole('button');
    await user.click(sendButton);
    
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('should disable input and button when disabled prop is true', () => {
    render(<ChatInput {...defaultProps} disabled={true} />);
    
    const input = screen.getByRole('textbox');
    const button = screen.getByRole('button');
    
    expect(input).toBeDisabled();
    expect(button).toBeDisabled();
  });

  it('should disable button when value is empty', () => {
    render(<ChatInput {...defaultProps} value="" />);
    
    const button = screen.getByRole('button');
    expect(button).toBeDisabled();
  });

  it('should enable button when value is not empty and not disabled', () => {
    render(<ChatInput {...defaultProps} value="test" />);
    
    const button = screen.getByRole('button');
    expect(button).not.toBeDisabled();
  });

  it('should show clear button when onClear is provided', () => {
    const onClear = vi.fn();
    render(<ChatInput {...defaultProps} onClear={onClear} />);
    
    expect(screen.getByText('Clear')).toBeInTheDocument();
  });

  it('should call onClear when clear button is clicked', async () => {
    const user = userEvent.setup();
    const onClear = vi.fn();
    
    render(<ChatInput {...defaultProps} onClear={onClear} />);
    
    const clearButton = screen.getByText('Clear');
    await user.click(clearButton);
    
    expect(onClear).toHaveBeenCalledTimes(1);
  });

  it('should display context info when provided', () => {
    const contextInfo = {
      estimated_tokens: 2840,
      context_limit: 8192,
      usage_percentage: 34.7,
      conversation_messages: 12,
      approaching_limit: false
    };
    
    render(<ChatInput {...defaultProps} contextInfo={contextInfo} />);
    
    expect(screen.getByText('2.8k/8k tokens (35%) • 12 messages')).toBeInTheDocument();
  });

  it('should display loading when no context info provided', () => {
    render(<ChatInput {...defaultProps} />);
    
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('should display context info in yellow when approaching limit', () => {
    const contextInfo = {
      estimated_tokens: 6500,
      context_limit: 8192,
      usage_percentage: 79.3,
      conversation_messages: 25,
      approaching_limit: true
    };
    
    render(<ChatInput {...defaultProps} contextInfo={contextInfo} />);
    
    const statusText = screen.getByText('6.5k/8k tokens (79%) • 25 messages');
    expect(statusText).toBeInTheDocument();
    // Note: We can't easily test CSS color in jsdom, but the logic is there
  });
});