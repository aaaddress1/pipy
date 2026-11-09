export class ToolError extends Error {
	constructor(message: string, readonly context?: Record<string, unknown>) {
		super(message);
		this.name = "ToolError";
	}

	render(): string {
		return this.message;
	}
}

export function isRecord(value: unknown): value is Record<string, unknown> {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function isEnoent(error: unknown): boolean {
	return error instanceof Error && (error as Error & { code?: string }).code === "ENOENT";
}

export const logger = {
	warn(message: string, detail?: Record<string, unknown>): void {
		process.stderr.write(message + (detail ? " " + JSON.stringify(detail) : "") + "\n");
	},
};
