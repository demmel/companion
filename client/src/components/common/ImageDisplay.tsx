import { css } from "@styled-system/css";

interface ImageDisplayProps {
  src: string;
  alt?: string;
  maxWidth?: string;
  maxHeight?: string;
  onClick?: () => void;
  exactSize?: boolean; // When true, use exact width/height; when false, use max constraints
}

export function ImageDisplay({
  src,
  alt = "Image",
  maxWidth = "200px",
  maxHeight = "150px",
  onClick,
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
      </div>
    );
  }
}
