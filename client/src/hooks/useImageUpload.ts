import { useState } from "react";
import { AgentClient } from "@/client";
import { ImageUploadResponse } from "@/types";

export interface UploadedImage extends ImageUploadResponse {
  preview?: string;
}

export interface UseImageUploadReturn {
  uploadedImages: UploadedImage[];
  uploading: boolean;
  uploadImages: (files: FileList | File[]) => Promise<void>;
  removeImage: (id: string) => void;
  clearImages: () => void;
}

export const useImageUpload = (client: AgentClient): UseImageUploadReturn => {
  const [uploadedImages, setUploadedImages] = useState<UploadedImage[]>([]);
  const [uploading, setUploading] = useState(false);

  const uploadImages = async (files: FileList | File[]) => {
    setUploading(true);
    try {
      const uploadPromises = Array.from(files).map(async (file) => {
        const result = await client.uploadImage(file);
        return {
          ...result,
          preview: URL.createObjectURL(file),
        };
      });
      const results = await Promise.all(uploadPromises);
      setUploadedImages((prev) => [...prev, ...results]);
    } catch (error) {
      console.error("Failed to upload images:", error);
      throw error;
    } finally {
      setUploading(false);
    }
  };

  const removeImage = (id: string) => {
    setUploadedImages((prev) => {
      const imageToRemove = prev.find((img) => img.id === id);
      if (imageToRemove?.preview) {
        URL.revokeObjectURL(imageToRemove.preview);
      }
      return prev.filter((img) => img.id !== id);
    });
  };

  const clearImages = () => {
    // Clean up object URLs
    uploadedImages.forEach((img) => {
      if (img.preview) {
        URL.revokeObjectURL(img.preview);
      }
    });
    setUploadedImages([]);
  };

  return {
    uploadedImages,
    uploading,
    uploadImages,
    removeImage,
    clearImages,
  };
};
