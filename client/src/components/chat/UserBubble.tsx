import { css } from '@styled-system/css';

interface UserBubbleProps {
  children: React.ReactNode;
  showHeader?: boolean;
}

export function UserBubble({ children, showHeader = false }: UserBubbleProps) {
  return (
    <div className={css({ mb: 4 })}>
      {showHeader && (
        <div className={css({ 
          mb: 3,
          display: 'flex',
          justifyContent: 'flex-end'
        })}>
          <span className={css({ 
            fontWeight: 'medium', 
            color: 'gray.300',
            fontSize: 'xl'
          })}>
            You
          </span>
        </div>
      )}
      
      <div className={css({ 
        display: 'flex', 
        justifyContent: 'flex-end' 
      })}>
        <div className={css({ 
          maxWidth: { base: 'xs', lg: 'md' }
        })}>
          <div className={css({ 
            bg: 'blue.600', 
            color: 'white', 
            rounded: '2xl',
            roundedTopRight: 'sm',
            px: 4, 
            py: 2 
          })}>
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}