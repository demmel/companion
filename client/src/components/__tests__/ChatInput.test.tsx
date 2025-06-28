import { describe, it, expect, vi } from 'vitest';
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
    const user = userEvent.setup();
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
    
    expect(screen.getByText('Clear conversation')).toBeInTheDocument();
  });

  it('should call onClear when clear button is clicked', async () => {
    const user = userEvent.setup();
    const onClear = vi.fn();
    
    render(<ChatInput {...defaultProps} onClear={onClear} />);
    
    const clearButton = screen.getByText('Clear conversation');
    await user.click(clearButton);
    
    expect(onClear).toHaveBeenCalledTimes(1);
  });

  it('should display item count and scroll mode', () => {
    render(<ChatInput {...defaultProps} itemCount={5} scrollMode="Manual scroll" />);
    
    expect(screen.getByText('5 items • Manual scroll')).toBeInTheDocument();
  });

  it('should display default values for item count and scroll mode', () => {
    render(<ChatInput {...defaultProps} />);
    
    expect(screen.getByText('0 items • Auto-scroll')).toBeInTheDocument();
  });
});