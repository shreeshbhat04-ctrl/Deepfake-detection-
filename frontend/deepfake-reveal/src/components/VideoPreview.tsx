import { useEffect, useRef } from "react";
import { Film } from "lucide-react";

interface VideoPreviewProps {
  videoUrl: string | null;
}

export const VideoPreview = ({ videoUrl }: VideoPreviewProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && videoUrl) {
      videoRef.current.load();
    }
  }, [videoUrl]);

  if (!videoUrl) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center bg-muted/30 rounded-2xl border border-border/50">
        <Film className="w-16 h-16 text-muted-foreground/50 mb-4" />
        <p className="text-muted-foreground">No video selected</p>
      </div>
    );
  }

  return (
    <div className="relative w-full h-full rounded-2xl overflow-hidden border border-border shadow-lg">
      <video
        ref={videoRef}
        controls
        className="w-full h-full object-contain bg-black"
      >
        <source src={videoUrl} type="video/mp4" />
        Your browser does not support the video tag.
      </video>
    </div>
  );
};
