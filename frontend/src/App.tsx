import { useEffect, useMemo } from "react";
import { Loader2, CheckCircle, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { PipelineCard } from "@/components/PipelineCard";
import { usePipelineStore } from "@/store/pipelineStore";

function App() {
  const categories = usePipelineStore((s) => s.categories);
  const loading = usePipelineStore((s) => s.loading);
  const fetchRegistry = usePipelineStore((s) => s.fetchRegistry);
  const fetchRuns = usePipelineStore((s) => s.fetchRuns);
  const pipelineStatus = usePipelineStore((s) => s.pipelineStatus);
  const runUpTo = usePipelineStore((s) => s.runUpTo);
  const runFullPipeline = usePipelineStore((s) => s.runFullPipeline);
  const pipelineError = usePipelineStore((s) => s.pipelineError);

  useEffect(() => {
    fetchRegistry();
    fetchRuns();
  }, [fetchRegistry, fetchRuns]);

  const oneTimeCategories = useMemo(
    () => categories.filter((c) => c.type === "one_time"),
    [categories],
  );
  const multiRunCategories = useMemo(
    () => categories.filter((c) => c.type === "multi_run"),
    [categories],
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <p className="text-muted-foreground">Loading plugin registry...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header bar */}
      <header className="border-b bg-white">
        <div className="container mx-auto flex items-center justify-between px-6 py-3">
          <h1 className="text-xl font-semibold tracking-tight">SARSSAe</h1>
          <a
            href="http://localhost:5000"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-muted-foreground hover:text-foreground"
          >
            <span className="italic">ml<span className="text-blue-500 font-semibold">flow</span></span>{" "}
            Pipeline Experiments Results
          </a>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        {/* Page title */}
        <h2 className="text-3xl font-semibold tracking-tight text-center mb-10">
          Run new pipeline experiment
        </h2>

        {/* One-time cards row */}
        {oneTimeCategories.length > 0 && (
          <section>
            <div className="grid gap-6" style={{
              gridTemplateColumns: `repeat(${oneTimeCategories.length}, minmax(0, 1fr))`,
            }}>
              {oneTimeCategories.map((cat) => (
                <PipelineCard
                  key={cat.name}
                  category={cat}
                  onRunUpTo={() => runUpTo(cat.name)}
                />
              ))}
            </div>
          </section>
        )}

        {/* Multi-run cards row */}
        {multiRunCategories.length > 0 && (
          <section className="mt-8">
            <Separator className="mb-6" />
            <div className="grid gap-6" style={{
              gridTemplateColumns: `repeat(${multiRunCategories.length}, minmax(0, 1fr))`,
            }}>
              {multiRunCategories.map((cat) => (
                <PipelineCard
                  key={cat.name}
                  category={cat}
                  onRunUpTo={() => runUpTo(cat.name)}
                />
              ))}
            </div>
          </section>
        )}

        {/* Pipeline status banner */}
        {pipelineStatus === "done" && (
          <div className="mt-6 flex items-center gap-2 rounded-lg bg-green-50 border border-green-200 px-4 py-3 text-sm text-green-700">
            <CheckCircle className="h-4 w-4 shrink-0" />
            Pipeline finished successfully. Check MLflow for results.
          </div>
        )}
        {pipelineStatus === "error" && pipelineError && (
          <div className="mt-6 flex items-start gap-2 rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700">
            <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
            <span className="break-all">{pipelineError}</span>
          </div>
        )}

        {/* Run full pipeline button */}
        <div className="mt-6">
          <Button
            className="w-full bg-blue-500 hover:bg-blue-600 text-white py-6 text-base font-medium cursor-pointer"
            disabled={pipelineStatus === "running"}
            onClick={() => runFullPipeline()}
          >
            {pipelineStatus === "running" ? (
              <><Loader2 className="h-5 w-5 mr-2 animate-spin" />Running pipeline...</>
            ) : (
              "Run full pipeline"
            )}
          </Button>
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t mt-12">
        <div className="container mx-auto px-6 py-6 flex items-center justify-center gap-8 text-sm text-muted-foreground">
          <span>&copy; {new Date().getFullYear()}</span>
          <a href="#" className="hover:text-foreground">Documentation</a>
          <a href="#" className="hover:text-foreground">Resources</a>
          <a href="#" className="hover:text-foreground">About</a>
        </div>
      </footer>
    </div>
  );
}

export default App
