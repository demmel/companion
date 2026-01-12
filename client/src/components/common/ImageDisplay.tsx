import { css } from "@styled-system/css";
import { RotateCw } from "lucide-react";

interface ImageDisplayProps {
  src: string;
  alt?: string;
  maxWidth?: string;
  maxHeight?: string;
  onClick?: () => void;
  onRegenerate?: () => void;
  exactSize?: boolean; // When true, use exact width/height; when false, use max constraints
}

export function ImageDisplay({
  src,
  alt = "Image",
  maxWidth = "200px",
  maxHeight = "150px",
  onClick,
  onRegenerate,
  exactSize = false,
}: ImageDisplayProps) {
  const handleClick = () => {
    if (onClick) {
      onClick();
    } else {
      // Default behavior: open in new tab
      window.open(src, "_blank");
    }
  };

  const handleRegenerate = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent image click
    if (onRegenerate) {
      onRegenerate();
    }
  };

  if (exactSize) {
    // For thumbnails - exact dimensions with cropping
    return (
      <div
        style={{
          width: maxWidth,
          height: maxHeight,
        }}
        className={css({
          position: "relative",
          bg: "gray.700",
          rounded: "md",
          overflow: "hidden",
          cursor: onClick ? "pointer" : "default",
          flexShrink: 0,
        })}
      >
        <img
          src={src}
          alt={alt}
          className={css({
            width: "100%",
            height: "100%",
            objectFit: "cover",
            display: "block",
          })}
          onClick={handleClick}
        />
        {onRegenerate && (
          <button
            onClick={handleRegenerate}
            className={css({
              position: "absolute",
              top: 2,
              right: 2,
              p: 1.5,
              bg: "rgba(0, 0, 0, 0.6)",
              rounded: "md",
              cursor: "pointer",
              transition: "background 0.2s",
              border: "none",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              _hover: { bg: "rgba(0, 0, 0, 0.8)" },
            })}
            aria-label="Regenerate image"
          >
            <RotateCw size={16} className={css({ color: "white" })} />
          </button>
        )}
      </div>
    );
  } else {
    // For full images - maintain aspect ratio within constraints
    return (
      <div
        className={css({
          position: "relative",
          maxWidth,
          bg: "gray.700",
          rounded: "md",
          overflow: "hidden",
          cursor: onClick ? "pointer" : "default",
          flexShrink: 0,
        })}
      >
        <img
          src={src}
          alt={alt}
          className={css({
            width: "100%",
            height: "auto",
            maxHeight,
            objectFit: "contain",
            display: "block",
          })}
          onClick={handleClick}
        />
        {onRegenerate && (
          <button
            onClick={handleRegenerate}
            className={css({
              position: "absolute",
              top: 2,
              right: 2,
              p: 1.5,
              bg: "rgba(0, 0, 0, 0.6)",
              rounded: "md",
              cursor: "pointer",
              transition: "background 0.2s",
              border: "none",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              _hover: { bg: "rgba(0, 0, 0, 0.8)" },
            })}
            aria-label="Regenerate image"
          >
            <RotateCw size={16} className={css({ color: "white" })} />
          </button>
        )}
      </div>
    );
  }
}
