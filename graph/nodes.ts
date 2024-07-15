import { AIMessage, BaseMessage } from "@langchain/core/messages";
import { renderedTools, tooList, toolMap } from "../tools";
import { AgentAction } from "@langchain/core/agents";
import {
  JsonOutputParser,
  StringOutputParser,
} from "@langchain/core/output_parsers";
import { createInterface } from "readline";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { model } from "../model";
import { llmSystemPrompt, prompt } from "../prompts";
import chalk from "chalk";

const readLineIns = createInterface({
  input: process.stdin,
  output: process.stdout,
});

export const INVOKE_MODEL = "invokeModel";
export const INVOKE_TOOL = "invokeTool";
export const USE_LLM = "useLLM";
export const NO_TOOLS = "noTools";
export const ASK_USER = "askUser";

const log = console.log;

// Define the function that determines whether to continue or not
/**
 * shouldContinue cannot be ASYNC, we need to create a new node after inital llm call
 * that will do this logic and than that will be connected to a shouldcontinue function
 */
export const shouldInvokeToolOrAskUser = async (state: {
  messages: Array<BaseMessage>;
}) => {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];

  log(
    chalk.cyan.bold(
      `shouldInvokeToolOrAskUser - Based on the last message determine what next agent to call`
    )
  );
  log(chalk.cyan(`Last Message: ${JSON.stringify(lastMessage)}`));

  // If there is no function name in the last message
  if (lastMessage.name && tooList.includes(lastMessage.name)) {
    log(chalk.blue.bold(`We need to run tool: ${lastMessage.name}`));
    // making it also a Promise since shouldInvokeToolOrAskUser is a async function
    // We also need to close the  readLineIns.close(); else after the final response the system does not return
    const answer = await new Promise<string>((resolve) => {
      readLineIns.close();
      resolve(INVOKE_TOOL);
    });
    return answer;
  } else {
    log(
      chalk.blue.bold(
        `Cannot find the answer using the available tools so asking users input`
      )
    );

    const answer = await new Promise<string>((resolve) => {
      readLineIns.question(
        "Sorry your question cannot be answered using the available tools. Do you want me to search web ? Type Y or N:  ",
        resolve
      );
    });

    if (!answer) {
      chalk.red(
        "User did not provide any answer, so ending graph and going to end node"
      );
      readLineIns.close();
      return NO_TOOLS;
    }
    if (answer.match(/^y(es)?$/i)) {
      chalk.green.bold("User answered as 'Yes', so using LLM to find answer");
      readLineIns.close();
      return USE_LLM;
    } else {
      chalk.magenta.bold(
        "User answered with 'No', so ending graph and going to end state"
      );
      readLineIns.close();
      return NO_TOOLS;
    }
  }
};

// Define the function to execute tools
const determineWhatToolToCall = (state: {
  messages: Array<BaseMessage>;
}): AgentAction => {
  const { messages } = state;
  // Based on the callTool response we know the last message involves a function call
  const lastMessage = messages[messages.length - 1];

  chalk.blue(
    `Pick the selected tool details from the list. Selected tool ${lastMessage.name}`
  );
  // We construct an AgentAction
  return {
    tool: lastMessage.name,
    toolInput: JSON.stringify(lastMessage.lc_kwargs.arguments),
    log: "",
  };
};

// Define the function that calls the model
export const callModel = async (state: { messages: Array<BaseMessage> }) => {
  const { messages } = state;
  const chain = prompt.pipe(model).pipe(new JsonOutputParser());

  log(chalk.blue.bold(`Invoking model to answer users question`));

  const response = await chain.invoke({
    input: messages[0].content,
    rendered_tools: renderedTools,
  });
  // We return a list, because this will get added to the existing list
  return {
    messages: [new AIMessage(response)],
  };
};

export const invokeTool = async (state: { messages: Array<BaseMessage> }) => {
  const action = determineWhatToolToCall(state);
  // We call the tool_executor and get back a response
  const chosenTool = toolMap[action.tool];
  const agruments = Object.values(JSON.parse(action.toolInput as string));

  log(chalk.cyan.bold(`Invoking tool: ${action.tool} to answer question`));

  const toolRes = await chosenTool.invoke({
    a: agruments[0],
    b: agruments[1],
  });

  log(chalk.cyan(`Tools response is:  ${toolRes}`));

  // We return a list, because this will get added to the existing list
  return { messages: [new AIMessage(toolRes)] };
};

export const useLLM = async (state: { messages: Array<BaseMessage> }) => {
  const { messages } = state;

  const chain = llmSystemPrompt.pipe(model).pipe(new StringOutputParser());
  const response = await chain.invoke({ question: messages[0].content });

  chalk.magenta(`LLM response after user intervention :  ${response}`);

  // We return a list, because this will get added to the existing list
  return {
    messages: [new AIMessage(response)],
  };
};

export const noTools = async (state: { messages: Array<BaseMessage> }) => {
  return {
    messages: [
      new AIMessage(
        `Sorry your question cannot be answered using the available tools, I have. I only have ${tooList.toString()} tools in my list `
      ),
    ],
  };
};
