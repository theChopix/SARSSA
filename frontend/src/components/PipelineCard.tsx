import { useMemo } from "react";
import { Settings, Check, Play, AlertCircle, Loader2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { PluginCategory, PluginImplementation } from "@/api/types";
import { usePipelineStore } from "@/store/pipelineStore";

interface PipelineCardProps {
  category: PluginCategory;
  onRunUpTo?: () => void;
}

export function PipelineCard({ category, onRunUpTo }: PipelineCardProps) {
  const card = usePipelineStore((s) => s.cards[category.name]);
  const previousRuns = usePipelineStore((s) => s.previousRuns);
  const setCardMode = usePipelineStore((s) => s.setCardMode);
  const selectPlugin = usePipelineStore((s) => s.selectPlugin);
  const setParam = usePipelineStore((s) => s.setParam);
  const toggleConfig = usePipelineStore((s) => s.toggleConfig);
  const loadFromRun = usePipelineStore((s) => s.loadFromRun);
  const setTargetRun = usePipelineStore((s) => s.setTargetRun);
  const doExecuteMultiRunStep = usePipelineStore((s) => s.executeMultiRunStep);

  const mode = card?.mode ?? "new";
  const selectedPlugin = card?.selectedPlugin ?? null;
  const configOpen = card?.configOpen ?? null;
  const params = card?.params ?? {};
  const loadedRunId = card?.loadedRunId ?? null;
  const targetRunId = card?.targetRunId ?? null;
  const cardStatus = card?.status ?? "idle";
  const executionLog = card?.executionLog ?? [];

  const isMultiRun = category.type === "multi_run";
  const cardError = card?.error ?? null;
  const isRunning = cardStatus === "running";

  const borderClass =
    cardStatus === "running"
      ? "border-blue-400"
      : cardStatus === "done"
        ? "border-green-400"
        : cardStatus === "error"
          ? "border-red-400"
          : "";

  const eligibleRuns = useMemo(
    () =>
      previousRuns.filter(
        (run) => run.status === "FINISHED" && run.context != null,
      ),
    [previousRuns],
  );

  const loadEligibleRuns = useMemo(
    () =>
      previousRuns.filter(
        (run) =>
          run.status === "FINISHED" &&
          run.context != null &&
          run.context[category.name] != null,
      ),
    [previousRuns, category.name],
  );

  const groups = useMemo(() => {
    const grouped = new Map<string, PluginImplementation[]>();
    for (const impl of category.implementations) {
      const key = impl.group ?? "__default__";
      const list = grouped.get(key) ?? [];
      list.push(impl);
      grouped.set(key, list);
    }
    return grouped;
  }, [category.implementations]);

  const hasGroups = groups.size > 1 || !groups.has("__default__");

  function getParamValue(pluginPath: string, paramName: string, defaultVal: unknown): string {
    const val = params[pluginPath]?.[paramName];
    if (val !== undefined && val !== null) return String(val);
    if (defaultVal !== undefined && defaultVal !== null) return String(defaultVal);
    return "";
  }

  function renderImplementation(impl: PluginImplementation) {
    const isSelected = selectedPlugin === impl.plugin_path;
    const isConfigOpen = configOpen === impl.plugin_path;

    return (
      <div key={impl.plugin_path} className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <RadioGroupItem value={impl.plugin_path} id={impl.plugin_path} />
            <Label
              htmlFor={impl.plugin_path}
              className="cursor-pointer font-normal"
            >
              {impl.display_name}
            </Label>
          </div>
          {impl.params.length > 0 && (
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                toggleConfig(category.name, impl.plugin_path);
              }}
              className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1 cursor-pointer"
            >
              <Settings className="h-3.5 w-3.5" />
              Configure
            </button>
          )}
        </div>

        {impl.params.length > 0 && isConfigOpen && isSelected && (
          <div className="ml-6 pl-2 border-l-2 border-blue-200 space-y-3 py-2">
            {impl.params.map((p) => (
              <div key={p.name} className="flex items-center gap-3">
                <Label className="text-xs text-muted-foreground w-24 shrink-0">
                  {p.name}
                  <span className="ml-1 text-[10px] opacity-60">({p.type})</span>
                </Label>
                <Input
                  type={p.type === "int" || p.type === "float" ? "number" : "text"}
                  step={p.type === "float" ? "any" : undefined}
                  value={getParamValue(impl.plugin_path, p.name, p.default)}
                  onChange={(e) =>
                    setParam(category.name, impl.plugin_path, p.name, e.target.value)
                  }
                  className="h-7 text-sm"
                />
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  if (isMultiRun) {
    return (
      <Card className={`flex flex-col ${borderClass}`}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">{category.display_name}</CardTitle>
            <StatusIndicator status={cardStatus} />
          </div>
        </CardHeader>

        <CardContent className="flex-1 flex flex-col gap-4">
          {/* Target run selector */}
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">
              Target pipeline run
            </Label>
            {eligibleRuns.length === 0 ? (
              <p className="text-sm text-muted-foreground italic py-2">
                Run a one-time pipeline first.
              </p>
            ) : (
              <Select
                value={targetRunId ?? undefined}
                onValueChange={(val) => setTargetRun(category.name, val as string)}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select a pipeline run..." />
                </SelectTrigger>
                <SelectContent>
                  {eligibleRuns.map((run) => (
                    <SelectItem key={run.run_id} value={run.run_id}>
                      {run.run_name}
                      <span className="ml-2 text-xs text-muted-foreground">
                        ({new Date(run.start_time).toLocaleString()})
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>

          <Separator />

          {/* Implementation radio list */}
          <RadioGroup
            value={selectedPlugin ?? ""}
            onValueChange={(val) => selectPlugin(category.name, val)}
            className="space-y-1"
          >
            {hasGroups
              ? Array.from(groups.entries()).map(([groupName, impls]) => (
                  <div key={groupName}>
                    {groupName !== "__default__" && (
                      <div className="flex items-center gap-2 mt-2 mb-1">
                        <Badge variant="outline" className="text-xs font-normal">
                          {groupName}
                        </Badge>
                        <Separator className="flex-1" />
                      </div>
                    )}
                    <div className="space-y-1">
                      {impls.map(renderImplementation)}
                    </div>
                  </div>
                ))
              : category.implementations.map(renderImplementation)}
          </RadioGroup>

          {/* Execute button */}
          <div className="mt-auto pt-2">
            <Button
              className="w-full bg-blue-500 hover:bg-blue-600 text-white cursor-pointer"
              disabled={!targetRunId || !selectedPlugin || cardStatus === "running"}
              onClick={() => doExecuteMultiRunStep(category.name)}
            >
              <Play className="h-4 w-4 mr-1" />
              {cardStatus === "running" ? "Executing..." : "Execute step"}
            </Button>
          </div>

          {/* Error message */}
          {cardStatus === "error" && cardError && (
            <div className="flex items-start gap-1.5 text-xs text-red-600 bg-red-50 rounded px-2 py-1.5">
              <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
              <span className="break-all">{cardError}</span>
            </div>
          )}

          {/* Execution log */}
          {executionLog.length > 0 && (
            <div className="space-y-1">
              <Label className="text-xs text-muted-foreground">Execution log</Label>
              <div className="max-h-32 overflow-y-auto space-y-1">
                {executionLog.map((entry, i) => (
                  <div
                    key={i}
                    className={`flex items-center gap-1.5 text-xs px-2 py-1 rounded ${
                      entry.status === "done"
                        ? "bg-green-50 text-green-700"
                        : "bg-red-50 text-red-700"
                    }`}
                  >
                    {entry.status === "done" ? (
                      <Check className="h-3 w-3 shrink-0" />
                    ) : (
                      <AlertCircle className="h-3 w-3 shrink-0" />
                    )}
                    <span className="truncate">
                      {entry.plugin} → {entry.runName}
                    </span>
                    <span className="ml-auto text-[10px] opacity-60 shrink-0">
                      {new Date(entry.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={`flex flex-col ${borderClass}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{category.display_name}</CardTitle>
          <StatusIndicator status={cardStatus} />
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col gap-4">
        {/* Mode toggle */}
        <div className="flex gap-0 rounded-md overflow-hidden border border-blue-500">
          <ModeButton
            label="Set up new"
            active={mode === "new"}
            onClick={() => setCardMode(category.name, "new")}
          />
          <ModeButton
            label="Load from previous run"
            active={mode === "load"}
            onClick={() => setCardMode(category.name, "load")}
          />
        </div>

        {/* New mode: implementation radio list */}
        {mode === "new" && (
          <RadioGroup
            value={selectedPlugin ?? ""}
            onValueChange={(val) => selectPlugin(category.name, val)}
            className="space-y-1"
          >
            {hasGroups
              ? Array.from(groups.entries()).map(([groupName, impls]) => (
                  <div key={groupName}>
                    {groupName !== "__default__" && (
                      <>
                        <div className="flex items-center gap-2 mt-2 mb-1">
                          <Badge variant="outline" className="text-xs font-normal">
                            {groupName}
                          </Badge>
                          <Separator className="flex-1" />
                        </div>
                      </>
                    )}
                    <div className="space-y-1">
                      {impls.map(renderImplementation)}
                    </div>
                  </div>
                ))
              : category.implementations.map(renderImplementation)}
          </RadioGroup>
        )}

        {/* Load mode: previous run selector */}
        {mode === "load" && (
          <div className="space-y-2">
            {loadEligibleRuns.length === 0 ? (
              <p className="text-sm text-muted-foreground italic py-4 text-center">
                No previous runs found for this step.
              </p>
            ) : (
              <>
                <Label className="text-xs text-muted-foreground">
                  Select a previous run
                </Label>
                <Select
                  value={loadedRunId ?? undefined}
                  onValueChange={(val) => loadFromRun(category.name, val as string)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Choose a run..." />
                  </SelectTrigger>
                  <SelectContent>
                    {loadEligibleRuns.map((run) => (
                      <SelectItem key={run.run_id} value={run.run_id}>
                        {run.run_name}
                        <span className="ml-2 text-xs text-muted-foreground">
                          ({new Date(run.start_time).toLocaleString()})
                        </span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {loadedRunId && (
                  <div className="flex items-center gap-1.5 text-sm text-green-600">
                    <Check className="h-4 w-4" />
                    Context loaded from previous run
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* Error message */}
        {cardStatus === "error" && cardError && (
          <div className="flex items-start gap-1.5 text-xs text-red-600 bg-red-50 rounded px-2 py-1.5">
            <AlertCircle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
            <span className="break-all">{cardError}</span>
          </div>
        )}

        {/* Run up to this step button */}
        <div className="mt-auto pt-2">
          <Button
            variant="outline"
            className="w-full border-blue-400 text-blue-600 hover:bg-blue-50 cursor-pointer"
            disabled={isRunning}
            onClick={onRunUpTo}
          >
            {isRunning ? (
              <><Loader2 className="h-4 w-4 mr-1 animate-spin" />Running...</>
            ) : (
              "Run up to this step"
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function StatusIndicator({ status }: { status: string }) {
  if (status === "running") {
    return <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />;
  }
  if (status === "done") {
    return (
      <div className="flex items-center gap-1 text-green-600">
        <Check className="h-5 w-5" />
      </div>
    );
  }
  if (status === "error") {
    return (
      <div className="flex items-center gap-1 text-red-500">
        <AlertCircle className="h-5 w-5" />
      </div>
    );
  }
  return null;
}

function ModeButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex-1 px-3 py-1.5 text-sm font-medium transition-colors cursor-pointer ${
        active
          ? "bg-blue-500 text-white"
          : "bg-white text-blue-500 hover:bg-blue-50"
      }`}
    >
      {label}
    </button>
  );
}
