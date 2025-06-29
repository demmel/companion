import { css } from '@styled-system/css';

interface StateHeaderProps {
  primaryText: string;
  icon?: string;
  secondaryText?: string;
}

export function StateHeader({ primaryText, icon, secondaryText }: StateHeaderProps) {
  return (
    <div className={css({ mb: 3 })}>
      <div className={css({ 
        display: 'flex', 
        alignItems: 'center', 
        gap: 2, 
        fontSize: 'sm' 
      })}>
        <span className={css({ 
          fontWeight: 'medium', 
          color: 'gray.300' 
        })}>
          {primaryText}
        </span>
        
        {icon && (
          <span className={css({ fontSize: 'lg' })}>
            {icon}
          </span>
        )}
        
        {secondaryText && (
          <span className={css({ 
            color: 'gray.500', 
            fontSize: 'xs' 
          })}>
            {secondaryText}
          </span>
        )}
      </div>
    </div>
  );
}