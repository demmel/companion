import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RoleplayText } from '../RoleplayText';

describe('RoleplayText rendering', () => {
  it('should render plain text correctly', () => {
    render(<RoleplayText content="Hello world" />);
    
    const element = screen.getByText('Hello world');
    expect(element).toBeInTheDocument();
    expect(element).toHaveClass('text-gray-900'); // default style
  });

  it('should render quoted text with correct styling', () => {
    render(<RoleplayText content='"Hello there"' />);
    
    const element = screen.getByText('"Hello there"');
    expect(element).toBeInTheDocument();
    expect(element).toHaveClass('text-green-600', 'font-medium');
  });

  it('should render action text with correct styling', () => {
    render(<RoleplayText content='*waves hand*' />);
    
    const element = screen.getByText('*waves hand*');
    expect(element).toBeInTheDocument();
    expect(element).toHaveClass('text-blue-500', 'italic');
  });

  it('should render thought text with correct styling', () => {
    render(<RoleplayText content='(thinking deeply)' />);
    
    const element = screen.getByText('(thinking deeply)');
    expect(element).toBeInTheDocument();
    expect(element).toHaveClass('text-yellow-600', 'opacity-75');
  });

  it('should render mixed content with multiple spans', () => {
    render(<RoleplayText content='She said "Hello" and *waved*' />);
    
    // Check that each part is rendered
    expect(screen.getByText(/She said/)).toBeInTheDocument();
    expect(screen.getByText('"Hello"')).toBeInTheDocument();
    expect(screen.getByText(/and/)).toBeInTheDocument();
    expect(screen.getByText('*waved*')).toBeInTheDocument();
    
    // Check styling
    expect(screen.getByText('"Hello"')).toHaveClass('text-green-600');
    expect(screen.getByText('*waved*')).toHaveClass('text-blue-500', 'italic');
  });

  it('should preserve whitespace with whitespace-pre-wrap', () => {
    const { container } = render(<RoleplayText content="Line 1\nLine 2\t\tTabbed" />);
    
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper).toHaveClass('whitespace-pre-wrap');
  });

  it('should apply custom className', () => {
    const { container } = render(
      <RoleplayText content="Test" className="custom-class" />
    );
    
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper).toHaveClass('custom-class');
  });

  it('should handle empty content gracefully', () => {
    const { container } = render(<RoleplayText content="" />);
    
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper).toBeInTheDocument();
    expect(wrapper).toBeEmptyDOMElement();
  });

  it('should handle complex nested scenarios correctly', () => {
    render(<RoleplayText content='Text "quoted *text*" more (thoughts) end' />);
    
    // Verify the quote section maintains quote styling throughout
    expect(screen.getByText('"quoted *text*"')).toHaveClass('text-green-600');
    
    // Verify thoughts are styled correctly
    expect(screen.getByText('(thoughts)')).toHaveClass('text-yellow-600');
    
    // Verify plain text using partial matches
    expect(screen.getByText(/Text/)).toBeInTheDocument();
    expect(screen.getByText(/more/)).toBeInTheDocument();
    expect(screen.getByText(/end/)).toBeInTheDocument();
  });

  it('should use consistent span structure for efficiency', () => {
    const { container } = render(<RoleplayText content='"Hello world"' />);
    
    // Should be a single span for the entire quoted text
    const spans = container.querySelectorAll('span span'); // nested spans
    expect(spans).toHaveLength(1); // Only one inner span
    expect(spans[0]).toHaveTextContent('"Hello world"');
  });
});