import { Loader2 } from "lucide-react";

export const AnalyzingState = () => {
  return (
    <div className="animate-slide-up rounded-2xl p-8 border border-primary/50 bg-card/50 backdrop-blur-sm">
      <div className="flex flex-col items-center space-y-6">
        <div className="relative">
          <div className="absolute inset-0 animate-ping rounded-full bg-primary/20" />
          <div className="relative p-6 rounded-full bg-primary/10 border border-primary/30">
            <Loader2 className="w-12 h-12 text-primary animate-spin" />
          </div>
        </div>
        
        <div className="text-center space-y-2">
          <h3 className="text-2xl font-bold font-display text-foreground">
            Analyzing Video...
          </h3>
          <p className="text-muted-foreground">
            Running deepfake detection algorithms
          </p>
        </div>

        <div className="w-full space-y-3">
          {[
            { text: "Extracting frames", delay: "0s" },
            { text: "Detecting faces", delay: "0.5s" },
            { text: "Running neural analysis", delay: "1s" },
          ].map((step, index) => (
            <div 
              key={index} 
              className="flex items-center gap-3 text-sm text-muted-foreground"
              style={{ animationDelay: step.delay }}
            >
              <div className="w-2 h-2 rounded-full bg-primary animate-pulse-glow" />
              <span>{step.text}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Scanner effect */}
      <div className="relative w-full h-1 mt-8 bg-muted/30 rounded-full overflow-hidden">
        <div className="scanner-line absolute inset-0" />
      </div>
    </div>
  );
};
