// Runs the example three-question request against a local server with @typesafe-ai/sdk and prints
// one JSON object. The client reads TYPESAFE_API_KEY / TYPESAFE_BASE_URL / TYPESAFE_DEFAULT_MODEL.
import { readFileSync } from "node:fs";
import { TypeSafeClient, choice, noul, score } from "@typesafe-ai/sdk";

const example = JSON.parse(readFileSync(new URL("../../examples/request.json", import.meta.url)));
const q = example.questions;
const client = new TypeSafeClient({ retry: { maxRetries: 0 } });

const caught = async (fn) => {
  try { await fn(); return null; } catch (e) { return { name: e.constructor.name, status: e.status }; }
};

const promise = client.systemOne({ state: example.state, questions: {
  route: choice(q.route.instructions, q.route.criteria),
  frustration: score(q.frustration.instructions, q.frustration.criteria),
  transfer_problem: noul(q.transfer_problem.instructions),
} });
const { data: result, requestId } = await promise.withResponse();
console.log(JSON.stringify({
  result,
  request_id: requestId ?? null,
  models: await client.models.list(),
  auth: await caught(() => new TypeSafeClient({ apiKey: "wrong", retry: { maxRetries: 0 } }).systemOne(
    { state: "x", questions: { q: noul("i") } })),
  invalid: await caught(() => client.systemOne({ state: "x", questions: { q: choice("i", {}) } })),
}));
