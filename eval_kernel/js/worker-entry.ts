import { parentPort } from "node:worker_threads";
import { WorkerCore } from "./worker-core";
import type { WorkerInbound } from "./worker-protocol";

if (!parentPort) throw new Error("JavaScript eval Worker has no parent port.");
const port = parentPort;

new WorkerCore({
	send: message => port.postMessage(message),
	onMessage: handler => {
		const receive = (message: WorkerInbound) => handler(message);
		port.on("message", receive);
		return () => port.off("message", receive);
	},
	close: () => {
		port.close();
		setTimeout(() => process.exit(0), 0);
	},
}, { mode: "isolated" });
