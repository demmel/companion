import { css } from '@styled-system/css';

interface SystemBubbleProps {
  children: React.ReactNode;
}

export function SystemBubble({ children }: SystemBubbleProps) {
  return (
    <div className={css({ 
      mb: 4,
      display: 'flex',
      justifyContent: 'center'
    })}>
      <div className={css({ 
        px: 4,
        py: 2,
        bg: 'gray.900',
        border: '1px solid',
        borderColor: 'gray.700',
        rounded: 'md',
        fontSize: 'sm',
        color: 'gray.400',
        fontStyle: 'italic',
        maxWidth: 'lg',
        textAlign: 'center'
      })}>
        {children}
      </div>
    </div>
  );
}