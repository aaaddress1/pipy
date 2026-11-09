import { createInterface } from "node:readline";
import { shadowSnapshotDigest, type ShadowSnapshot } from "./shared/runtime";
import { projectJavaScriptShadowPlan } from "./speculation";
import { evaluateShadowExpression } from "./shadow/evaluator";
import type { ShadowToolCall, ShadowValue } from "./shadow/types";

type Request =
	| { type: "plan"; id: string; code: string; snapshot: ShadowSnapshot }
	| { type: "arguments"; id: string; values: Record<string, unknown>; results: Record<string, unknown>; calls: ShadowToolCall[] };

function resolveArguments(calls: ShadowToolCall[], snapshot: Readonly<Record<string, unknown>>, results: Map<string, ShadowValue>) {
	const args = [];
	for (const call of calls) {
		try {
			args.push([call.id, evaluateShadowExpression(call.args, { snapshot, results }).value]);
		} catch {
			// A missing value or an unsupported transformation declines this operation.
		}
	}
	return args;
}

const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of input) {
	const request = JSON.parse(line) as Request;
	try {
		if (request.type === "plan") {
			const snapshot = request.snapshot;
			const plan = await projectJavaScriptShadowPlan(request.code, {
				snapshot: snapshot.values,
				initialGlobals: snapshot.initialGlobals,
			});
			process.stdout.write(JSON.stringify({
				id: request.id,
				revision: snapshot.revision,
				digest: shadowSnapshotDigest(snapshot),
				...plan,
				args: resolveArguments(plan.operations.map(operation => operation.call).filter(call =>
					call.dependencies.length === 0 && call.controlDependencies.length === 0
				), snapshot.values, new Map()),
			}) + "\n");
		} else if (request.type === "arguments") {
			const results = new Map<string, ShadowValue>();
			for (const [id, value] of Object.entries(request.results)) results.set(id, { value, origins: [] });
			const args = resolveArguments(request.calls, request.values, results);
			process.stdout.write(JSON.stringify({ id: request.id, args }) + "\n");
		}
	} catch {
		process.stdout.write(JSON.stringify({ id: request.id, failed: true }) + "\n");
	}
}
