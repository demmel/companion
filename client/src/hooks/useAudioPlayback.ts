import { useState, useRef, useCallback } from "react";

export type PlayState = "idle" | "loading" | "playing";

interface UseAudioPlaybackResult {
  playState: PlayState;
  play: (audioUrl: string) => Promise<void>;
  stop: () => void;
  setLoading: () => void;
  setIdle: () => void;
}

export function useAudioPlayback(): UseAudioPlaybackResult {
  const [playState, setPlayState] = useState<PlayState>("idle");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const urlRef = useRef<string | null>(null);

  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    if (urlRef.current) {
      URL.revokeObjectURL(urlRef.current);
      urlRef.current = null;
    }
    setPlayState("idle");
  }, []);

  const play = useCallback(async (audioUrl: string) => {
    // Clean up any existing audio
    stop();

    const audio = new Audio(audioUrl);
    audioRef.current = audio;
    urlRef.current = audioUrl;

    audio.onended = () => {
      setPlayState("idle");
      if (urlRef.current) {
        URL.revokeObjectURL(urlRef.current);
        urlRef.current = null;
      }
    };

    audio.onerror = () => {
      setPlayState("idle");
      if (urlRef.current) {
        URL.revokeObjectURL(urlRef.current);
        urlRef.current = null;
      }
    };

    await audio.play();
    setPlayState("playing");
  }, [stop]);

  const setLoading = useCallback(() => setPlayState("loading"), []);
  const setIdle = useCallback(() => setPlayState("idle"), []);

  return { playState, play, stop, setLoading, setIdle };
}
