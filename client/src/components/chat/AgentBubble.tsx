import { css } from '@styled-system/css';

interface AgentBubbleProps {
  children: React.ReactNode;
}

export function AgentBubble({ children }: AgentBubbleProps) {
  return (
    <div className={css({ mb: 4 })}>
      <div className={css({ 
        bg: 'gray.800', 
        border: '1px solid', 
        borderColor: 'gray.700', 
        rounded: 'lg', 
        px: 4, 
        py: 2, 
        mb: 2 
      })}>
        {children}
      </div>
    </div>
  );
}