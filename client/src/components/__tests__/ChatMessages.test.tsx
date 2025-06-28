import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ChatMessages } from '../ChatMessages';

describe('ChatMessages', () => {
  const mockTextItem = {
    data: { type: 'text' as const, content: 'Hello world' }
  };

  const mockToolItem = {
    data: { 
      type: 'tool' as const, 
      toolId: 'tool1',
      name: 'search',
      parameters: { query: 'test' },
      status: 'running' as const
    }
  };

  it('should render empty state when no items', () => {
    render(<ChatMessages items={[]} />);
    
    expect(screen.getByText('Start a conversation with the agent!')).toBeInTheDocument();
    expect(screen.getByText('Try: "Please roleplay as Elena, a mysterious vampire."')).toBeInTheDocument();
  });

  it('should render text items', () => {
    render(<ChatMessages items={[mockTextItem]} />);
    
    expect(screen.getByText('Hello world')).toBeInTheDocument();
  });

  it('should render tool items', () => {
    render(<ChatMessages items={[mockToolItem]} />);
    
    expect(screen.getByText('search')).toBeInTheDocument();
  });

  it('should render multiple mixed items', () => {
    const items = [mockTextItem, mockToolItem];
    render(<ChatMessages items={items} />);
    
    expect(screen.getByText('Hello world')).toBeInTheDocument();
    expect(screen.getByText('search')).toBeInTheDocument();
  });

  it('should show streaming cursor when not complete and has items', () => {
    render(<ChatMessages items={[mockTextItem]} isComplete={false} />);
    
    const cursor = screen.getByText('▋');
    expect(cursor).toBeInTheDocument();
    expect(cursor).toHaveClass('animate-pulse', 'text-gray-500');
  });

  it('should not show streaming cursor when complete', () => {
    render(<ChatMessages items={[mockTextItem]} isComplete={true} />);
    
    expect(screen.queryByText('▋')).not.toBeInTheDocument();
  });

  it('should not show streaming cursor when no items', () => {
    render(<ChatMessages items={[]} isComplete={false} />);
    
    expect(screen.queryByText('▋')).not.toBeInTheDocument();
  });

  it('should call onScroll when scrolled', () => {
    const onScroll = vi.fn();
    const { container } = render(
      <ChatMessages items={[mockTextItem]} onScroll={onScroll} />
    );
    
    const scrollContainer = container.firstChild as HTMLElement;
    fireEvent.scroll(scrollContainer);
    
    expect(onScroll).toHaveBeenCalledTimes(1);
  });

  it('should apply custom className', () => {
    const { container } = render(
      <ChatMessages items={[]} className="custom-class" />
    );
    
    const scrollContainer = container.firstChild as HTMLElement;
    expect(scrollContainer).toHaveClass('custom-class');
  });

  it('should apply default classes', () => {
    const { container } = render(<ChatMessages items={[]} />);
    
    const scrollContainer = container.firstChild as HTMLElement;
    expect(scrollContainer).toHaveClass(
      'flex-1', 
      'overflow-y-auto', 
      'px-4', 
      'py-4', 
      'space-y-4'
    );
  });

  it('should forward ref correctly', () => {
    const ref = vi.fn();
    render(<ChatMessages ref={ref} items={[]} />);
    
    expect(ref).toHaveBeenCalledWith(expect.any(HTMLDivElement));
  });

  it('should render items with correct prose styling', () => {
    render(<ChatMessages items={[mockTextItem]} />);
    
    const itemContainer = screen.getByText('Hello world').closest('div');
    expect(itemContainer).toHaveClass('prose', 'prose-sm', 'max-w-none');
  });

  it('should render streaming cursor with correct prose styling', () => {
    render(<ChatMessages items={[mockTextItem]} isComplete={false} />);
    
    const cursorContainer = screen.getByText('▋').closest('div');
    expect(cursorContainer).toHaveClass('prose', 'prose-sm', 'max-w-none');
  });
});