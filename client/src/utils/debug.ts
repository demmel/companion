/**
 * Simple debug logging utility that only logs in development mode
 */

const isDev = import.meta.env.DEV;

export const debug = {
  log: (message: string, ...args: any[]) => {
    if (isDev) console.log(message, ...args);
  },

  warn: (message: string, ...args: any[]) => {
    if (isDev) console.warn(message, ...args);
  },

  error: (message: string, ...args: any[]) => {
    if (isDev) console.error(message, ...args);
  },

  info: (message: string, ...args: any[]) => {
    if (isDev) console.info(message, ...args);
  },

  group: (name: string) => {
    if (isDev) console.group(name);
  },

  groupEnd: () => {
    if (isDev) console.groupEnd();
  },
};
