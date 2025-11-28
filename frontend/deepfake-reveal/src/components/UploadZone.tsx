import { useCallback, useState } from "react";
import { Upload, Video } from "lucide-react";
import { cn } from "@/lib/utils";

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  isAnalyzing: boolean;
}

export const UploadZone = ({ onFileSelect, isAnalyzing }: UploadZoneProps) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true);
    }
  }, []);

  const handleDragOut = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        if (file.type.startsWith("video/")) {
          onFileSelect(file);
        }
      }
    },
    [onFileSelect]
  );

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div
      className={cn(
        "relative border-2 border-dashed rounded-2xl p-12 transition-all duration-300",
        isDragging
          ? "border-primary bg-primary/10 scale-[1.02]"
          : "border-border hover:border-primary/50 bg-card/50",
        isAnalyzing && "opacity-50 pointer-events-none"
      )}
      onDragEnter={handleDragIn}
      onDragLeave={handleDragOut}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input
        type="file"
        accept="video/*"
        onChange={handleFileInput}
        className="hidden"
        id="video-upload"
        disabled={isAnalyzing}
      />
      
      <label
        htmlFor="video-upload"
        className="flex flex-col items-center justify-center cursor-pointer space-y-4"
      >
        <div className={cn(
          "p-6 rounded-full transition-all duration-300",
          isDragging 
            ? "bg-primary/20 scale-110" 
            : "bg-muted/50"
        )}>
          {isDragging ? (
            <Video className="w-12 h-12 text-primary animate-bounce" />
          ) : (
            <Upload className="w-12 h-12 text-muted-foreground" />
          )}
        </div>
        
        <div className="text-center space-y-2">
          <p className="text-lg font-semibold text-foreground">
            {isDragging ? "Drop video here" : "Upload Video"}
          </p>
          <p className="text-sm text-muted-foreground">
            Drag and drop or click to select
          </p>
          <p className="text-xs text-muted-foreground">
            Supports MP4, MOV, AVI formats
          </p>
        </div>
      </label>
    </div>
  );
};
