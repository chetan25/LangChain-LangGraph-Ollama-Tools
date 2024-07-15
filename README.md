---
Title: LangChain + LangGraph + Ollama + Tools
Excerpt: Simple example showing how to use Tools with Ollama and LangChain and how to implement a human in the loop with LangGraph.
Tech: "Ollama, NodeJS, Typescript, LangGraph, LangChain"
---

# LangChain + LangGraph + Ollama + Tools

Using tools with OpenAI chat models were pretty simple since it supported function calling. We could just `bind` the available functions to the model and the model figures out to call the right function, w/o any prompting. Here is an example showing how to call tools with ChatOpenAI model:

```js
// This is from the Langchian example section
const calculatorSchema = z.object({
  operation: z
    .enum(["add", "subtract", "multiply", "divide"])
    .describe("The type of operation to execute."),
  number1: z.number().describe("The first number to operate on."),
  number2: z.number().describe("The second number to operate on."),
});

const calculatorTool = tool(
  async ({ operation, number1, number2 }) => {
    // Functions must return strings
    if (operation === "add") {
      return `${number1 + number2}`;
    } else if (operation === "subtract") {
      return `${number1 - number2}`;
    } else if (operation === "multiply") {
      return `${number1 * number2}`;
    } else if (operation === "divide") {
      return `${number1 / number2}`;
    } else {
      throw new Error("Invalid operation.");
    }
  },
  {
    name: "calculator",
    description: "Can perform mathematical operations.",
    schema: calculatorSchema,
  }
);

const llmWithTools = llm.bindTools([calculatorTool]);
```

With Ollama we need to nudge the model a little bit with some prompting. The basic idea of creating the `Tools` remains the same. We craft our prompt guiding the model that it has tools available at its disposal to answer the questions.

## Ollama with Tools

A full example of Ollama with tools is done in [ollama-tool.ts](./ollama-tools.ts) file. Let's break down the steps here:

1. First we create the tools we need, in the code below we are creating a tool called `addTool`. We can create tools with two ways:

   1. Either by calling the `tool` function - This provides a simple way of creating a tool function where we can omit few things and the function creates default values for us. Internally it also calls the `DynamicStructuredTool`.
   2. Or by instantiating a new instance of `DynamicStructuredTool` class and passing the provided values.

   ```js
   const adderSchema = z.object({
     a: z.number(),
     b: z.number(),
   });

   // using the tool function
   const addTool = tool(
     async (input): Promise<string> => {
       const sum = input.a + input.b;
       return `The sum of ${input.a} and ${input.b} is ${sum}`;
     },
     // { name: "adder" } // only works if input is a primitive vlaue
     {
       name: "adder",
       description: "This tool is use to add(plus) two numbers together",
       schema: adderSchema,
     }
   );

   // using the DynamicStructuredTool class
   const multiplyTool = new DynamicStructuredTool({
     name: "multiply",
     description: "This tool is used to multiply(into) two numbers together",
     schema: z.object({
       a: z.number().describe("the first number to multiply"),
       b: z.number().describe("the second number to multiply"),
     }),
     func: async ({ a, b }: { a: number, b: number }) => {
       return (a * b).toString();
     },
   });
   ```

2. Now we create a system prompt, that will guide the model on the available tools and when to use them. We cannot pass the tools directly to the prompt, so first we create a text descriptions of all the tools present, as the prompt can only take string values.

   ```js
   const tools = [addTool, multiplyTool];
   const renderedTools = renderTextDescription(tools);

   // now we pass this description to the prompt
   const systemPrompt = `You are a helpful assistant that has access to the following set of tools to answer users question.
   Here are the names and descriptions for each tool:
   {rendered_tools}
   Given the user's input, return the name and input of the tool that can be used to
   accurately and correctly ....
   `;
   ```

3. We now create a final prompt which is an instance of ChatPromptTemplate with the above System prompt and users question as input.

   ```js
   const prompt = ChatPromptTemplate.fromMessages([
     ["system", systemPrompt],
     ["user", "{input}"],
   ]);
   ```

4. The final part is invoking the model. If we only invoke the model by piping in the prompt and the output formatter, we will only get the response as what tool to called with the arguments. Example shown below:
   ```js
   // This will return a json object with the tool name and arguments only
   const response = await prompt
     .pipe(model)
     .pipe(new JsonOutputParser())
     .invoke({
       input: question,
       rendered_tools: renderedTools,
     });
   ```
5. But we need to also invoke the tool so we pipe the output of the model to a `Runnable` sequence called `RunnablePick` and `RunnableLambda`. The `RunnablePick` is used to pick the selective data attribute from the previous message and pass it to the next function call. `RunnableLambda` basically takes a function and makes it a runnable function.

```js
  const chain = prompt
    .pipe(model)
    .pipe(new JsonOutputParser())
    .pipe(
      new RunnableLambda({
        func: (input: { name: string; arguments: {} }) => {
          // create an Object with tool aname as key and value as the fn
          const toolMap: Record<string, StructuredToolInterface> =
            Object.fromEntries(tools.map((tool) => [tool.name, tool]));
          const choosenTool = toolMap[input.name];
          const agruments = Object.values(input.arguments);
          return choosenTool.invoke({
            a: agruments[0],
            b: agruments[1],
          });
        },
      })
```

## Ollama with LangGraph(Human in loop)

We can build amazing autonomous agents with LangChain using super amazing library called LangGraph. State based transitions could not be simplified better than what we have in LangGraph. I have a full example of showing how to build a autonomous agent with human intervention in [ollama-tool-graph.ts](./ollama-tool-graph.ts). Let's break down the steps here:

1. The flow for creating the tools is exactly the same as what we did earlier. Even the prompt creation is the same. We still guide the model about the available tools and when to use them. But the invocation of the model is slightly different.
2. We first start by defining our `Nodes` and `Edges`. Nodes basically are functions that we want to run when system is on specific state. Edges are basically path connecting the two nodes and it tells the system to transition from one node to the next one.
3. After this we decide what our Graph state would look like, for us we only want to store the previous messages as our nodes would react to those.
   ```js
   interface IState {
     messages: BaseMessage[];
   }
   const graphState: StateGraphArgs<IState>["channels"] = {
     messages: {
       value: (x: BaseMessage[], y: BaseMessage[]) => x.concat(y),
       default: () => [],
     },
   };
   ```
4. Now we can define our graph and add the nodes to it:

   ```js
   const workflow = new StateGraph({
     channels: graphState,
   });

   // Define the nodes we will cycle between
   workflow.addNode(INVOKE_MODEL, callModel);
   workflow.addNode(INVOKE_TOOL, invokeTool);
   workflow.addNode(USE_LLM, useLLM);
   workflow.addNode(NO_TOOLS, noTools);
   ```

5. Here is the list of different nodes and a short description on what they do:

   1. callModel - This function takes in the current state and plucks the user question of it and invokes the model. It also returns the new state with the model response added as an instance of `AIMessage` to it. Its the start node of our graph.
   2. invokeTool - This function take in the current state and determines what tool to call. It also invokes the model determined tool and returns new updated state with the tool response added as an instance of `AIMessage` to it.
   3. useLLM - This function take in the current state and plucks the original question and uses a model with a simple prompt and no tools. It also returns the new state with the model response added as an instance of `AIMessage` to it
   4. noTools - This function basically just returns a new state with the a "Sorry Response" added as an instance of `AIMessage` to it.

6. Now we define the conditional edge and the edges between the nodes

   ```js
   workflow.addConditionalEdges(INVOKE_MODEL, shouldInvokeToolOrAskUser, {
     invokeTool: INVOKE_TOOL,
     noTools: NO_TOOLS,
     useLLM: USE_LLM,
   });

   workflow.addEdge(INVOKE_TOOL, END);
   workflow.addEdge(NO_TOOLS, END);
   workflow.addEdge(USE_LLM, END);
   ```

7. The `shouldInvokeToolOrAskUser` function is the place where the model system decides if it can answer the users questions using the tool or does it need human intervention.Based on the current state if there is a tool name in the State message we return `INVOKE_TOOL` but if not then we use the command line to take user input and based on that return either `NO_TOOLS` or `USE_LLM`. The last state for every node is the `END` state.
   ```js
   // Human intervention
   const answer =
     (await new Promise()) <
     string >
     ((resolve) => {
       readLineIns.question(
         "Sorry your question cannot be answered using the available tools. Do you want me to search web ? Type Y or N:  ",
         resolve
       );
     });
   if (!answer) {
     readLineIns.close();
     return NO_TOOLS;
   }
   if (answer.match(/^y(es)?$/i)) {
     readLineIns.close();
     return USE_LLM;
   } else {
     readLineIns.close();
     return NO_TOOLS;
   }
   ```

## Local Development

For running it locally we need to:

1. Make sure we have local `Ollama` running or replace the ollama chat model with OpenAI one.
2. Install the dependencies
   ```js
   npm install
   ```
3. To run we have two commands
   ```js
   "start:graph": "npx tsx ollama-tool-graph.ts ",
   "start:tool": "npx tsx ollama-tools.ts"
   ```
