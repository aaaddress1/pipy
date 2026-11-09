import { createConnection } from "node:net";
import { createInterface } from "node:readline";
import { startJsEvalProcess } from "./process-entry";
import type { WorkerInbound, WorkerOutbound } from "./worker-protocol";

const encode = JSON.stringify.bind(JSON);
const decode = JSON.parse.bind(JSON);
const setPrototype = Object.setPrototypeOf.bind(Object);
const port = Number(process.env.MINI_PI_EVAL_PORT);
const token = process.env.MINI_PI_EVAL_TOKEN;
const threaded = process.env.MINI_PI_EVAL_THREAD === "1";
delete process.env.MINI_PI_EVAL_PORT;
delete process.env.MINI_PI_EVAL_TOKEN;
delete process.env.MINI_PI_EVAL_THREAD;
if (!token || !Number.isSafeInteger(port) || port < 1 || port > 65535) {
	throw new Error("Missing Mini Pi eval worker connection.");
}

const socket = createConnection({ host: "127.0.0.1", port });
socket.setNoDelay(true);
const input = createInterface({ input: socket, crlfDelay: Infinity });
socket.on("connect", () => socket.write(token + "\n"));
socket.on("error", () => process.exit(1));
socket.on("close", () => process.exit(0));

const transport = {
	send(message: WorkerOutbound) {
		socket.write(encode(message.type === "shadow-snapshot" ? setPrototype(message, null) : message) + "\n");
	},
	onMessage(handler: (message: WorkerInbound) => void) {
		const receive = (line: string) => handler(decode(line) as WorkerInbound);
		input.on("line", receive);
		return () => input.off("line", receive);
	},
};

if (threaded) {
	const worker = new Worker(new URL("./worker-entry.ts", import.meta.url).href, { type: "module" });
	transport.onMessage(message => worker.postMessage(message));
	worker.addEventListener("message", event => transport.send(event.data));
	worker.addEventListener("error", () => socket.destroy());
	worker.addEventListener("close", () => socket.end());
} else {
	startJsEvalProcess(transport, consume => {
		const rejected = (reason: unknown) => {
			if (!consume(reason)) setTimeout(() => { throw reason; }, 0);
		};
		process.on("unhandledRejection", rejected);
		return () => process.off("unhandledRejection", rejected);
	});
}
