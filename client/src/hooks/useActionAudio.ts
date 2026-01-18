import { useCallback, useRef } from "react";
import { useAudioPlayback, PlayState } from "./useAudioPlayback";

interface UseActionAudioOptions {
  triggerId: string;
  actionIndex: number;
}

interface UseActionAudioResult {
  playState: PlayState;
  handlePlayClick: () => void;
}

export function useActionAudio({
  triggerId,
  actionIndex,
}: UseActionAudioOptions): UseActionAudioResult {
  const { playState, play, stop, setLoading, setIdle } = useAudioPlayback();
  const retryTimeoutRef = useRef<number | null>(null);

  const cancelRetry = useCallback(() => {
    if (retryTimeoutRef.current) {
      window.clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }
  }, []);

  const handlePlayClick = useCallback(async () => {
    // If playing, stop
    if (playState === "playing") {
      stop();
      return;
    }

    // If loading, cancel
    if (playState === "loading") {
      cancelRetry();
      setIdle();
      return;
    }

    // Start loading
    setLoading();
    const audioUrl = `/api/audio/${triggerId}/${actionIndex}`;

    const tryFetch = async (): Promise<void> => {
      try {
        const response = await fetch(audioUrl);

        if (response.status === 202) {
          // Not ready yet, retry after 1 second
          retryTimeoutRef.current = window.setTimeout(() => {
            tryFetch();
          }, 1000);
          return;
        }

        if (response.status === 200) {
          // Audio is ready, play it
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          await play(url);
          return;
        }

        // Other status codes - give up
        console.error(`Audio fetch failed with status ${response.status}`);
        setIdle();
      } catch (error) {
        console.error("Failed to fetch audio:", error);
        setIdle();
      }
    };

    tryFetch();
  }, [playState, triggerId, actionIndex, play, stop, setLoading, setIdle, cancelRetry]);

  return { playState, handlePlayClick };
}
