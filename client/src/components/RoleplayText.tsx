import { css } from '@styled-system/css';

interface TextSpan {
  content: string;
  style: 'default' | 'quote' | 'action' | 'thought';
}

interface RoleplayTextProps {
  content: string;
  className?: string;
}

export function parseRoleplayText(content: string): TextSpan[] {
  const spans: TextSpan[] = [];
  let currentSpan: TextSpan = { content: '', style: 'default' };
  
  // State machine for formatting
  let inQuotes = false;
  let inAsterisks = false; 
  let inParens = false;

  for (const char of content) {
    // Determine what style this character should have
    let charStyle: TextSpan['style'];
    
    if (char === '"') {
      charStyle = 'quote';
      inQuotes = !inQuotes; // Toggle after determining style
    } else if (char === '*' && !inQuotes) {
      charStyle = 'action'; 
      inAsterisks = !inAsterisks;
    } else if (char === '(' && !inParens && !inQuotes) {
      charStyle = 'thought';
      inParens = true;
    } else if (char === ')' && inParens) {
      charStyle = 'thought';
      inParens = false;
    } else {
      // Regular character - use current formatting state
      charStyle = inQuotes ? 'quote' : 
                  inAsterisks ? 'action' : 
                  inParens ? 'thought' : 'default';
    }
    
    // Only create new span if style changed
    if (charStyle !== currentSpan.style) {
      if (currentSpan.content) {
        spans.push({ ...currentSpan });
      }
      currentSpan = { content: '', style: charStyle };
    }
    
    // Always append current character
    currentSpan.content += char;
  }

  // Add final span
  if (currentSpan.content) {
    spans.push(currentSpan);
  }

  return spans;
}

function getSpanStyles(style: TextSpan['style']) {
  switch (style) {
    case 'quote':
      return css({ color: 'green.400', fontWeight: 'medium' });
    case 'action':
      return css({ color: 'blue.400', fontStyle: 'italic' });  
    case 'thought':
      return css({ color: 'yellow.400', opacity: 0.75 });
    default:
      return css({ color: 'gray.100' }); // default text color for dark theme
  }
}

export function RoleplayText({ content, className = '' }: RoleplayTextProps) {
  const spans = parseRoleplayText(content);
  
  const baseClass = css({ whiteSpace: 'pre-wrap' });
  const combinedClass = className ? `${baseClass} ${className}` : baseClass;
  
  return (
    <span className={combinedClass}>
      {spans.map((span, index) => (
        <span 
          key={index} 
          className={getSpanStyles(span.style)}
        >
          {span.content}
        </span>
      ))}
    </span>
  );
}