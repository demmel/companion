import React, { useState, useRef, useEffect } from 'react';
import { Check, X, User } from 'lucide-react';
import { css } from '@styled-system/css';
import { useUsername } from '@/contexts/UsernameContext';

interface UsernameSettingsProps {
  isOpen: boolean;
  onClose: () => void;
}

export function UsernameSettings({ isOpen, onClose }: UsernameSettingsProps) {
  const { username, setUsername } = useUsername();
  const [inputValue, setInputValue] = useState(username);
  const [isEditing, setIsEditing] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Update input value when username changes
  useEffect(() => {
    setInputValue(username);
  }, [username]);

  // Focus input when starting to edit
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  // Reset when closing
  useEffect(() => {
    if (!isOpen) {
      setIsEditing(false);
      setInputValue(username);
    }
  }, [isOpen, username]);

  // Close when clicking escape
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      return () => document.removeEventListener('keydown', handleKeyDown);
    }
  }, [isOpen, onClose]);

  const handleSave = () => {
    const trimmedValue = inputValue.trim();
    if (trimmedValue) {
      setUsername(trimmedValue);
      setIsEditing(false);
    } else {
      // Reset to current username if empty
      setInputValue(username);
      setIsEditing(false);
    }
  };

  const handleCancel = () => {
    setInputValue(username);
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSave();
    } else if (e.key === 'Escape') {
      handleCancel();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className={css({
        position: 'absolute',
        top: '100%',
        left: 0,
        mt: 2,
        bg: 'gray.800',
        border: '1px solid',
        borderColor: 'gray.600',
        rounded: 'lg',
        p: 4,
        minWidth: '280px',
        zIndex: 50,
        boxShadow: 'lg',
      })}
    >
      <div
        className={css({
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          mb: 3,
        })}
      >
        <div
          className={css({
            display: 'flex',
            alignItems: 'center',
            gap: 2,
          })}
        >
          <User size={16} className={css({ color: 'gray.400' })} />
          <h3
            className={css({
              fontSize: 'sm',
              fontWeight: 'medium',
              color: 'white',
            })}
          >
            Username
          </h3>
        </div>
        <button
          onClick={onClose}
          className={css({
            p: 1,
            color: 'gray.400',
            _hover: { color: 'white' },
            transition: 'colors',
            rounded: 'md',
          })}
        >
          <X size={14} />
        </button>
      </div>

      <div
        className={css({
          display: 'flex',
          alignItems: 'center',
          gap: 2,
        })}
      >
        {isEditing ? (
          <>
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter username..."
              maxLength={50}
              className={css({
                flex: 1,
                px: 2,
                py: 1,
                bg: 'gray.700',
                border: '1px solid',
                borderColor: 'gray.600',
                rounded: 'md',
                color: 'white',
                fontSize: 'sm',
                _placeholder: { color: 'gray.400' },
                _focus: {
                  outline: 'none',
                  borderColor: 'blue.500',
                },
              })}
            />
            <button
              onClick={handleSave}
              disabled={!inputValue.trim()}
              className={css({
                p: 1,
                color: 'green.400',
                _hover: { color: 'green.300' },
                _disabled: {
                  color: 'gray.600',
                  cursor: 'not-allowed',
                },
                transition: 'colors',
              })}
            >
              <Check size={16} />
            </button>
            <button
              onClick={handleCancel}
              className={css({
                p: 1,
                color: 'red.400',
                _hover: { color: 'red.300' },
                transition: 'colors',
              })}
            >
              <X size={16} />
            </button>
          </>
        ) : (
          <>
            <span
              className={css({
                flex: 1,
                color: 'gray.300',
                fontSize: 'sm',
              })}
            >
              {username}
            </span>
            <button
              onClick={() => setIsEditing(true)}
              className={css({
                px: 2,
                py: 1,
                bg: 'blue.600',
                color: 'white',
                rounded: 'md',
                fontSize: 'xs',
                fontWeight: 'medium',
                _hover: { bg: 'blue.700' },
                transition: 'colors',
              })}
            >
              Edit
            </button>
          </>
        )}
      </div>

      <p
        className={css({
          fontSize: 'xs',
          color: 'gray.500',
          mt: 2,
        })}
      >
        This name will appear in chat messages from you.
      </p>
    </div>
  );
}