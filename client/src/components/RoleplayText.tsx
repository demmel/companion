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
  let currentSpan = { content: '', style: 'default' as const };
  
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
      currentSpan = { content: '', style: charStyle as TextSpan['style'] };
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

function getSpanClassName(style: TextSpan['style']): string {
  switch (style) {
    case 'quote':
      return 'text-green-600 font-medium'; // bright_green equivalent
    case 'action':
      return 'text-blue-500 italic'; // italic bright_blue equivalent  
    case 'thought':
      return 'text-yellow-600 opacity-75'; // dim yellow equivalent
    default:
      return 'text-gray-900'; // default text color
  }
}

export function RoleplayText({ content, className = '' }: RoleplayTextProps) {
  const spans = parseRoleplayText(content);
  
  return (
    <span className={`whitespace-pre-wrap ${className}`}>
      {spans.map((span, index) => (
        <span 
          key={index} 
          className={getSpanClassName(span.style)}
        >
          {span.content}
        </span>
      ))}
    </span>
  );
}