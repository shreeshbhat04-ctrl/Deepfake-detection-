import { Shield, AlertTriangle } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface AnalysisResultProps {
  prediction: "real" | "fake";
  confidence: number;
}

export const AnalysisResult = ({ prediction, confidence }: AnalysisResultProps) => {
  const isReal = prediction === "real";

  return (
    <div className={cn(
      "animate-slide-up rounded-2xl p-8 border-2 transition-all duration-500",
      isReal 
        ? "bg-primary/10 border-primary glow-real" 
        : "bg-destructive/10 border-destructive glow-fake"
    )}>
      <div className="flex items-center gap-4 mb-6">
        <div className={cn(
          "p-4 rounded-full animate-pulse-glow",
          isReal ? "bg-primary/20" : "bg-destructive/20"
        )}>
          {isReal ? (
            <Shield className={cn("w-10 h-10", "text-primary")} />
          ) : (
            <AlertTriangle className={cn("w-10 h-10", "text-destructive")} />
          )}
        </div>
        
        <div className="flex-1">
          <h3 className={cn(
            "text-3xl font-bold font-display mb-1",
            isReal ? "text-primary" : "text-destructive"
          )}>
            {isReal ? "AUTHENTIC" : "DEEPFAKE DETECTED"}
          </h3>
          <p className="text-muted-foreground">
            {isReal 
              ? "This video appears to be genuine" 
              : "This video may be AI-generated or manipulated"}
          </p>
        </div>
      </div>

      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium text-foreground">Confidence Level</span>
          <span className={cn(
            "text-2xl font-bold font-display",
            isReal ? "text-primary" : "text-destructive"
          )}>
            {confidence.toFixed(1)}%
          </span>
        </div>
        
        <Progress 
          value={confidence} 
          className={cn(
            "h-3",
            isReal ? "[&>div]:bg-primary" : "[&>div]:bg-destructive"
          )}
        />
        
        <div className="flex justify-between text-xs text-muted-foreground mt-2">
          <span>Low</span>
          <span>Medium</span>
          <span>High</span>
        </div>
      </div>

      <div className={cn(
        "mt-6 p-4 rounded-lg border text-sm",
        isReal 
          ? "bg-primary/5 border-primary/30" 
          : "bg-destructive/5 border-destructive/30"
      )}>
        <p className="text-foreground/90">
          {isReal 
            ? "✓ No signs of digital manipulation detected. The video passed our deepfake detection analysis."
            : "⚠ Warning: Multiple indicators suggest this video may have been artificially generated or modified using AI technology."}
        </p>
      </div>
    </div>
  );
};
