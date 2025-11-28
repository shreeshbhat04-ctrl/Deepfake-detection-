import { useState } from "react";
import { Shield, Zap, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { UploadZone } from "@/components/UploadZone";
import { VideoPreview } from "@/components/VideoPreview";
import { AnalysisResult } from "@/components/AnalysisResult";
import { AnalyzingState } from "@/components/AnalyzingState";
import { toast } from "sonner";
import heroBackground from "@/assets/hero-background.jpg";

const Index = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<{
    prediction: "real" | "fake";
    confidence: number;
  } | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setResult(null);
    
    // Create object URL for video preview
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    
    toast.success("Video loaded successfully");
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      toast.error("Please upload a video first");
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    // Simulate API call to your Python backend
    // Replace this with actual API call to your Streamlit/Python backend
    setTimeout(() => {
      // Mock result - replace with actual API response
      const mockResult = {
        prediction: Math.random() > 0.5 ? "real" : "fake",
        confidence: Math.random() * 30 + 70, // 70-100%
      } as const;

      setResult(mockResult);
      setIsAnalyzing(false);
      
      toast.success("Analysis complete!");
    }, 4000);

    // TODO: Replace with actual API call:
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    const response = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      body: formData,
    });
    
    const data = await response.json();
    setResult({
      prediction: data.prediction,
      confidence: data.confidence,
    });
    setIsAnalyzing(false);
    
  };

  const handleReset = () => {
    setSelectedFile(null);
    setVideoUrl(null);
    setResult(null);
    setIsAnalyzing(false);
    
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div 
        className="relative min-h-screen flex flex-col"
        style={{
          backgroundImage: `url(${heroBackground})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundAttachment: 'fixed',
        }}
      >
        {/* Overlay */}
        <div className="absolute inset-0 bg-background/90 backdrop-blur-sm" />

        {/* Content */}
        <div className="relative z-10 flex-1">
          {/* Header */}
          <header className="border-b border-border/50 backdrop-blur-md bg-background/50">
            <div className="container mx-auto px-6 py-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-primary/10 border border-primary/30">
                    <Shield className="w-6 h-6 text-primary" />
                  </div>
                  <h1 className="text-2xl font-bold font-display text-foreground">
                    SafeGuard AI
                  </h1>
                </div>
                
              
              </div>
            </div>
          </header>

          {/* Main Content */}
          <main className="container mx-auto px-6 py-12">
            <div className="max-w-7xl mx-auto">
              {/* Title Section */}
              <div className="text-center mb-12 space-y-4">
                <h2 className="text-5xl md:text-6xl font-bold font-display text-foreground leading-tight">
                  Deepfake Detection
                  <span className="block text-primary mt-2">Powered by AI</span>
                </h2>
                <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                  Advanced neural network analysis to detect AI-generated and manipulated videos
                </p>
                
                {/* Feature Pills */}
                <div className="flex flex-wrap justify-center gap-4 pt-6">
                  {[
                    { icon: Zap, text: "Real-time Analysis" },
                    { icon: Eye, text: "High Accuracy" },
                    { icon: Shield, text: "Secure Processing" },
                  ].map((feature, index) => (
                    <div 
                      key={index}
                      className="flex items-center gap-2 px-4 py-2 rounded-full bg-muted/50 border border-border/50"
                    >
                      <feature.icon className="w-4 h-4 text-primary" />
                      <span className="text-sm font-medium text-foreground">{feature.text}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Main Interface */}
              <div className="grid lg:grid-cols-2 gap-8 mb-8">
                {/* Left Column - Upload */}
                <div className="space-y-6">
                  <div className="bg-card/80 backdrop-blur-sm rounded-2xl border border-border p-6 shadow-xl">
                    <h3 className="text-xl font-semibold font-display mb-4 text-foreground">
                      1. Upload Video
                    </h3>
                    <UploadZone 
                      onFileSelect={handleFileSelect} 
                      isAnalyzing={isAnalyzing}
                    />
                    
                    {selectedFile && (
                      <div className="mt-4 flex items-center justify-between p-4 bg-muted/30 rounded-lg">
                        <div className="flex items-center gap-3">
                          <div className="p-2 rounded bg-primary/10">
                            <Shield className="w-4 h-4 text-primary" />
                          </div>
                          <div>
                            <p className="text-sm font-medium text-foreground">
                              {selectedFile.name}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                            </p>
                          </div>
                        </div>
                        <Button 
                          variant="ghost" 
                          size="sm"
                          onClick={handleReset}
                          disabled={isAnalyzing}
                        >
                          Remove
                        </Button>
                      </div>
                    )}
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-4">
                    <Button
                      onClick={handleAnalyze}
                      disabled={!selectedFile || isAnalyzing}
                      className="flex-1 h-14 text-lg font-semibold bg-primary hover:bg-primary/90 text-primary-foreground shadow-lg hover:shadow-xl transition-all duration-300"
                      size="lg"
                    >
                      {isAnalyzing ? "Analyzing..." : "Analyze Video"}
                    </Button>
                    
                    {(result || isAnalyzing) && (
                      <Button
                        onClick={handleReset}
                        variant="outline"
                        className="h-14"
                        size="lg"
                        disabled={isAnalyzing}
                      >
                        New Scan
                      </Button>
                    )}
                  </div>
                </div>

                {/* Right Column - Preview & Results */}
                <div className="space-y-6">
                  <div className="bg-card/80 backdrop-blur-sm rounded-2xl border border-border p-6 shadow-xl">
                    <h3 className="text-xl font-semibold font-display mb-4 text-foreground">
                      2. Video Preview
                    </h3>
                    <div className="aspect-video">
                      <VideoPreview videoUrl={videoUrl} />
                    </div>
                  </div>
                </div>
              </div>

              {/* Results Section */}
              {(isAnalyzing || result) && (
                <div className="max-w-2xl mx-auto">
                  {isAnalyzing ? (
                    <AnalyzingState />
                  ) : result ? (
                    <AnalysisResult 
                      prediction={result.prediction}
                      confidence={result.confidence}
                    />
                  ) : null}
                </div>
              )}
            </div>
          </main>

          {/* Footer */}
          <footer className="border-t border-border/50 backdrop-blur-md bg-background/50 mt-auto">
            <div className="container mx-auto px-6 py-6">
              <p className="text-center text-sm text-muted-foreground">
                Built with DeepGuard AI • Advanced Deepfake Detection Technology
              </p>
            </div>
          </footer>
        </div>
      </div>
    </div>
  );
};

export default Index;
