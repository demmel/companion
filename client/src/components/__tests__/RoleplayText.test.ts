import { describe, it, expect } from 'vitest';
import { parseRoleplayText } from '../RoleplayText';

describe('parseRoleplayText', () => {
  it('should handle plain text', () => {
    const result = parseRoleplayText('Hello world');
    expect(result).toEqual([
      { content: 'Hello world', style: 'default' }
    ]);
  });

  it('should handle simple quotes', () => {
    const result = parseRoleplayText('"Hello"');
    expect(result).toEqual([
      { content: '"Hello"', style: 'quote' }
    ]);
  });

  it('should handle quotes with surrounding text', () => {
    const result = parseRoleplayText('She said "Hello" to me');
    expect(result).toEqual([
      { content: 'She said ', style: 'default' },
      { content: '"Hello"', style: 'quote' },
      { content: ' to me', style: 'default' }
    ]);
  });

  it('should handle simple actions', () => {
    const result = parseRoleplayText('*waves hand*');
    expect(result).toEqual([
      { content: '*waves hand*', style: 'action' }
    ]);
  });

  it('should handle actions with surrounding text', () => {
    const result = parseRoleplayText('Hello *waves* there');
    expect(result).toEqual([
      { content: 'Hello ', style: 'default' },
      { content: '*waves*', style: 'action' },
      { content: ' there', style: 'default' }
    ]);
  });

  it('should handle simple thoughts', () => {
    const result = parseRoleplayText('(thinking about dinner)');
    expect(result).toEqual([
      { content: '(thinking about dinner)', style: 'thought' }
    ]);
  });

  it('should handle thoughts with surrounding text', () => {
    const result = parseRoleplayText('Well (hmm) I think so');
    expect(result).toEqual([
      { content: 'Well ', style: 'default' },
      { content: '(hmm)', style: 'thought' },
      { content: ' I think so', style: 'default' }
    ]);
  });

  it('should prioritize quotes over actions', () => {
    const result = parseRoleplayText('"I *really* mean it"');
    expect(result).toEqual([
      { content: '"I *really* mean it"', style: 'quote' }
    ]);
  });

  it('should handle multiple quotes', () => {
    const result = parseRoleplayText('"Hello" and "Goodbye"');
    expect(result).toEqual([
      { content: '"Hello"', style: 'quote' },
      { content: ' and ', style: 'default' },
      { content: '"Goodbye"', style: 'quote' }
    ]);
  });

  it('should handle multiple actions', () => {
    const result = parseRoleplayText('*jumps* and *runs*');
    expect(result).toEqual([
      { content: '*jumps*', style: 'action' },
      { content: ' and ', style: 'default' },
      { content: '*runs*', style: 'action' }
    ]);
  });

  it('should handle multiple thoughts', () => {
    const result = parseRoleplayText('(first thought) and (second thought)');
    expect(result).toEqual([
      { content: '(first thought)', style: 'thought' },
      { content: ' and ', style: 'default' },
      { content: '(second thought)', style: 'thought' }
    ]);
  });

  it('should handle complex mixed formatting', () => {
    const result = parseRoleplayText('She said "Hello" *waves* (thinking) and left');
    expect(result).toEqual([
      { content: 'She said ', style: 'default' },
      { content: '"Hello"', style: 'quote' },
      { content: ' ', style: 'default' },
      { content: '*waves*', style: 'action' },
      { content: ' ', style: 'default' },
      { content: '(thinking)', style: 'thought' },
      { content: ' and left', style: 'default' }
    ]);
  });

  it('should handle unmatched quotes', () => {
    const result = parseRoleplayText('She said "Hello world');
    expect(result).toEqual([
      { content: 'She said ', style: 'default' },
      { content: '"Hello world', style: 'quote' }
    ]);
  });

  it('should handle unmatched actions', () => {
    const result = parseRoleplayText('She *started to wave');
    expect(result).toEqual([
      { content: 'She ', style: 'default' },
      { content: '*started to wave', style: 'action' }
    ]);
  });

  it('should handle unmatched thoughts', () => {
    const result = parseRoleplayText('Hmm (I wonder');
    expect(result).toEqual([
      { content: 'Hmm ', style: 'default' },
      { content: '(I wonder', style: 'thought' }
    ]);
  });

  it('should handle parentheses inside quotes (no thought formatting)', () => {
    const result = parseRoleplayText('"Hello (there) friend"');
    expect(result).toEqual([
      { content: '"Hello (there) friend"', style: 'quote' }
    ]);
  });

  it('should handle asterisks inside quotes (no action formatting)', () => {
    const result = parseRoleplayText('"The *best* day ever"');
    expect(result).toEqual([
      { content: '"The *best* day ever"', style: 'quote' }
    ]);
  });

  it('should handle empty string', () => {
    const result = parseRoleplayText('');
    expect(result).toEqual([]);
  });

  it('should handle only formatting characters', () => {
    const result = parseRoleplayText('""**()');
    expect(result).toEqual([
      { content: '""', style: 'quote' },
      { content: '**', style: 'action' },
      { content: '()', style: 'thought' }
    ]);
  });

  it('should not create unnecessary spans for efficiency', () => {
    // This tests the key efficiency improvement
    const result = parseRoleplayText('"Hello"');
    
    // Should be one span, not three (opening quote + content + closing quote)
    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({ content: '"Hello"', style: 'quote' });
  });

  it('should handle adjacent different formatting types efficiently', () => {
    const result = parseRoleplayText('"Hello"*waves*');
    
    // Should be exactly two spans
    expect(result).toHaveLength(2);
    expect(result[0]).toEqual({ content: '"Hello"', style: 'quote' });
    expect(result[1]).toEqual({ content: '*waves*', style: 'action' });
  });
});