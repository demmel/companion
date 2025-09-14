import { useState, useRef, useEffect } from "react";
import { Settings, Menu } from "lucide-react";
import { css } from "@styled-system/css";
import { AutoWakeupToggle } from "@/components/AutoWakeupToggle";
import { UsernameSettings } from "@/components/UsernameSettings";
import { AgentClient } from "@/client";

interface ChatHeaderProps {
  title?: string;
  isConnected?: boolean;
  isConnecting?: boolean;
  client: AgentClient;
}

export function ChatHeader({ client }: ChatHeaderProps) {
  const [showMenu, setShowMenu] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu and settings when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowMenu(false);
        setShowSettings(false);
      }
    };

    if (showMenu || showSettings) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showMenu, showSettings]);

  return (
    <div
      className={css({
        bg: "gray.800",
        borderBottom: "1px solid",
        borderColor: "gray.600",
        px: 4,
        py: 3,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      })}
    >
      <div
        className={css({
          display: "flex",
          alignItems: "center",
          gap: 3,
        })}
      >
        <div
          ref={menuRef}
          className={css({
            position: "relative",
          })}
        >
          <button
            onClick={() => setShowMenu(!showMenu)}
            className={css({
              p: 2,
              color: "gray.400",
              _hover: { color: "white", bg: "gray.700" },
              transition: "colors",
              rounded: "md",
            })}
          >
            <Menu size={20} />
          </button>

          {showMenu && (
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
                py: 2,
                minWidth: '180px',
                zIndex: 50,
                boxShadow: 'lg',
              })}
            >
              <button
                onClick={() => {
                  setShowSettings(true);
                  setShowMenu(false);
                }}
                className={css({
                  w: 'full',
                  px: 4,
                  py: 2,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 2,
                  color: 'gray.300',
                  _hover: { bg: 'gray.700', color: 'white' },
                  transition: 'colors',
                  textAlign: 'left',
                })}
              >
                <Settings size={16} />
                Settings
              </button>
            </div>
          )}

          <UsernameSettings
            isOpen={showSettings}
            onClose={() => setShowSettings(false)}
          />
        </div>
      </div>
      <div
        className={css({
          display: "flex",
          alignItems: "center",
          gap: 3,
        })}
      >
        <AutoWakeupToggle client={client} />
      </div>
    </div>
  );
}
