import { Plus } from "lucide-react";
import { css } from "@styled-system/css";
import { useRef } from "react";

interface FileUploadButtonProps {
  uploading: boolean;
  onUpload: (files: FileList | File[]) => Promise<void>;
  disabled?: boolean;
}

export function FileUploadButton({
  uploading,
  onUpload,
  disabled = false,
}: FileUploadButtonProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const files = event.target.files;
    if (!files) return;

    try {
      await onUpload(files);
    } finally {
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <>
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*"
        onChange={handleFileSelect}
        style={{ display: "none" }}
      />

      {/* Compact upload button */}
      <button
        type="button"
        onClick={handleUploadClick}
        disabled={disabled || uploading}
        className={css({
          p: 2,
          bg: "gray.600",
          color: "white",
          rounded: "md",
          width: "32px",
          height: "32px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          _hover: { bg: "gray.700" },
          _focus: { outline: "none" },
          _disabled: {
            bg: "gray.700",
            cursor: "not-allowed",
          },
          transition: "colors",
        })}
      >
        <Plus size={16} />
      </button>
    </>
  );
}
