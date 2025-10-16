import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";

interface UsernameContextType {
  username: string;
  setUsername: (username: string) => void;
}

const UsernameContext = createContext<UsernameContextType | undefined>(
  undefined,
);

export const useUsername = (): UsernameContextType => {
  const context = useContext(UsernameContext);
  if (!context) {
    throw new Error("useUsername must be used within a UsernameProvider");
  }
  return context;
};

interface UsernameProviderProps {
  children: ReactNode;
}

export const UsernameProvider: React.FC<UsernameProviderProps> = ({
  children,
}) => {
  const [username, setUsernameState] = useState<string>("User");

  // Load username from localStorage on mount
  useEffect(() => {
    const storedUsername = localStorage.getItem("agent-username");
    if (storedUsername && storedUsername.trim()) {
      setUsernameState(storedUsername.trim());
    }
  }, []);

  // Update localStorage whenever username changes
  const setUsername = (newUsername: string) => {
    const trimmedUsername = newUsername.trim();
    if (trimmedUsername) {
      setUsernameState(trimmedUsername);
      localStorage.setItem("agent-username", trimmedUsername);
    } else {
      // If empty, reset to default
      setUsernameState("User");
      localStorage.setItem("agent-username", "User");
    }
  };

  return (
    <UsernameContext.Provider value={{ username, setUsername }}>
      {children}
    </UsernameContext.Provider>
  );
};
