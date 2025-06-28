import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ChatHeader } from '../ChatHeader';

describe('ChatHeader', () => {
  it('should render default title and disconnected state', () => {
    render(<ChatHeader />);
    
    expect(screen.getByText('Agent Chat')).toBeInTheDocument();
    expect(screen.getByText('Disconnected')).toBeInTheDocument();
    expect(screen.getByText('Disconnected')).toHaveClass('text-red-500');
  });

  it('should render custom title', () => {
    render(<ChatHeader title="Custom Chat" />);
    
    expect(screen.getByText('Custom Chat')).toBeInTheDocument();
  });

  it('should show connecting state', () => {
    render(<ChatHeader isConnecting={true} />);
    
    expect(screen.getByText('Connecting...')).toBeInTheDocument();
    expect(screen.getByText('Connecting...')).toHaveClass('text-yellow-500');
  });

  it('should show connected state', () => {
    render(<ChatHeader isConnected={true} />);
    
    expect(screen.getByText('Connected')).toBeInTheDocument();
    expect(screen.getByText('Connected')).toHaveClass('text-green-500');
  });

  it('should prioritize connecting state over connected state', () => {
    render(<ChatHeader isConnected={true} isConnecting={true} />);
    
    expect(screen.getByText('Connecting...')).toBeInTheDocument();
    expect(screen.queryByText('Connected')).not.toBeInTheDocument();
  });
});