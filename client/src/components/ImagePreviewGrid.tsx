import { css } from "@styled-system/css";
import { X } from "lucide-react";
import { ImageDisplay } from "./common/ImageDisplay";
import { UploadedImage } from "@/hooks/useImageUpload";

interface ImagePreviewGridProps {
  images: UploadedImage[];
  onRemove: (id: string) => void;
}

export function ImagePreviewGrid({ images, onRemove }: ImagePreviewGridProps) {
  if (images.length === 0) {
    return null;
  }

  return (
    <div
      className={css({
        mb: 4,
        display: "flex",
        flexWrap: "wrap",
        gap: 2,
      })}
    >
      {images.map((image) => (
        <div
          key={image.id}
          className={css({
            position: "relative",
          })}
        >
          <ImageDisplay
            src={image.preview || image.url}
            alt={`Uploaded image ${image.id}`}
            maxWidth="60px"
            maxHeight="60px"
            exactSize={true}
            onClick={() => {}} // Disable click to open
          />
          <button
            type="button"
            onClick={() => onRemove(image.id)}
            className={css({
              position: "absolute",
              top: -1,
              right: -1,
              color: "white",
              bg: "rgba(0, 0, 0, 0.8)",
              _hover: { bg: "rgba(255, 0, 0, 0.8)" },
              p: "2px",
              rounded: "full",
              _focus: { outline: "none" },
              fontSize: "xs",
              width: "18px",
              height: "18px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            })}
          >
            <X size={12} />
          </button>
        </div>
      ))}
    </div>
  );
}
