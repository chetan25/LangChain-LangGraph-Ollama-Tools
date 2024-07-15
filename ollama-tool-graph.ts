import "dotenv/config";
import { HumanMessage } from "@langchain/core/messages";
import { END, START, StateGraph } from "@langchain/langgraph";
import { createWriteStream } from "fs";
import chalk from "chalk";
import { graphState } from "./graph";
import {
  callModel,
  invokeTool,
  INVOKE_MODEL,
  INVOKE_TOOL,
  NO_TOOLS,
  noTools,
  USE_LLM,
  useLLM,
  shouldInvokeToolOrAskUser,
} from "./graph/nodes";

const log = console.log;

async function init(question) {
  log(chalk.white.bgMagenta.bold("Ollama with Basic Tools and State Graph"));
  // Define a new graph
  const workflow = new StateGraph({
    channels: graphState,
  });

  // Define the nodes we will cycle between
  workflow.addNode(INVOKE_MODEL, callModel);
  workflow.addNode(INVOKE_TOOL, invokeTool);
  workflow.addNode(USE_LLM, useLLM);
  workflow.addNode(NO_TOOLS, noTools);

  // We now add a normal edge from `tools` to `agent`.
  // This means that after `tools` is called, `agent` node is called next.
  // @ts-ignore
  workflow.addConditionalEdges(INVOKE_MODEL, shouldInvokeToolOrAskUser, {
    invokeTool: INVOKE_TOOL,
    noTools: NO_TOOLS,
    useLLM: USE_LLM,
  });

  // workflow.addEdge("agent", "action");
  //@ts-ignore
  workflow.addEdge(INVOKE_TOOL, END);
  //@ts-ignore
  workflow.addEdge(NO_TOOLS, END);
  //@ts-ignore
  workflow.addEdge(USE_LLM, END);

  // Set the entrypoint as `agent`
  // This means that this node is the first one called
  //@ts-ignore
  workflow.addEdge(START, INVOKE_MODEL);

  // Finally, we compile it! This compiles it into a LangChain Runnable,
  const app = workflow.compile();

  const inputs = {
    messages: [new HumanMessage(question)],
  };
  const res = await app.invoke(inputs);
  log(chalk.bgMagenta.white.bold(`The final result from the LLM is: `));
  log(chalk.blue(`Question:  ${question} `));
  log(
    chalk.blue(
      `The final result from the LLM is: ${
        res.messages[res.messages.length - 1].content
      }`
    )
  );

  const blob = await app.getGraph().drawMermaidPng();
  const imageName = "public/images/graph.png";
  createWriteStream(imageName).write(Buffer.from(await blob.arrayBuffer()));
}

init("what is 2 plus 2");
