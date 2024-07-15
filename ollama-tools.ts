import "dotenv/config";
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { z } from "zod";
import {
  DynamicStructuredTool,
  StructuredToolInterface,
  tool,
} from "@langchain/core/tools";
import { renderTextDescription } from "langchain/tools/render";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { JsonOutputParser } from "@langchain/core/output_parsers";
import { RunnableLambda, RunnablePick } from "@langchain/core/runnables";
import chalk from "chalk";

const log = console.log;

async function init(question: string) {
  log(chalk.white.bgMagenta.bold("Ollama With Basic Tools"));
  // create the model instance, mistral works the best
  const model = new ChatOllama({ model: "mistral" });

  // creating schema for add tool
  const adderSchema = z.object({
    a: z.number(),
    b: z.number(),
  });

  /**
   * defining a add tool by
   * using the tool function which returns a StructuredTool instance
   *
   * The tool fn takes in 2 args, function to invoke and options objec which should have
   * name and  desciption and schema are optional which it troes to generate if not given.
   * It basically calls the DynamicStructuredTool and passes it the correct arguments
   **/
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

  /**
   * In this second way of creating a tool we are using DynamicStructuredTool
   */
  const multiplyTool = new DynamicStructuredTool({
    name: "multiply",
    description: "This tool is used to multiply(into) two numbers together",
    schema: z.object({
      a: z.number().describe("the first number to multiply"),
      b: z.number().describe("the second number to multiply"),
    }),
    func: async ({ a, b }: { a: number; b: number }) => {
      return (a * b).toString();
    },
  });

  const tools = [addTool, multiplyTool];
  const renderedTools = renderTextDescription(tools);

  log(chalk.blue("Tools Description"));

  log(chalk.blue.underline.bold(renderedTools));

  const systemPrompt = `You are a helpful assistant that has access to the following set of tools
  to answer users question.
  Here are the names and descriptions for each tool:
  {rendered_tools}
  
  Given the user's input, return the name and input of the tool that can be used to
  accurately and correctly answers users question without 
  any further clarification or assumption and should be exact match for the tool we are provided.
  If the users input can not be answered correctly using the provided tools and would need more 
  clarification or data, please don't
  make any assumptions just return name of tool name as "NA" with empty arguments as response.
  Return your response as a JSON blob with 'name' and 'arguments' keys.
  The value associated with the 'arguments' key should be a dictionary of parameters.
  `;

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", systemPrompt],
    ["user", "{input}"],
  ]);

  // If we just want to see that the LLM retuns what tool to call w/o callig it
  // const response = await prompt
  //   .pipe(model)
  //   .pipe(new JsonOutputParser())
  //   .invoke({
  //     input: question,
  //     rendered_tools: renderedTools,
  //   });
  // console.log({ response });

  const chain = prompt
    .pipe(model)
    .pipe(new JsonOutputParser())
    // RunnablePick basically picks the passed on keys from the input and returns them in an object as key value pair
    .pipe(new RunnablePick(["name", "arguments"]))
    .pipe(
      //  custom functions used as Runnables are called RunnableLambdas.
      // Note that all inputs to these functions need to be a SINGLE argument. If you have a function that
      // accepts multiple arguments, you should write a wrapper that accepts a single {} as input and unpacks it into multiple argument.
      new RunnableLambda({
        func: (input: { name: string; arguments: {} }) => {
          console.log({ input });
          log(
            chalk.blue(
              "LLM initial response telling what tool to call with what arguments: " +
                chalk.magenta.underline.bold(JSON.stringify(input))
            )
          );
          // create an Object with tool aname as key and value as the fn
          const toolMap: Record<string, StructuredToolInterface> =
            Object.fromEntries(tools.map((tool) => [tool.name, tool]));

          const choosenTool = toolMap[input.name];
          if (!choosenTool) {
            log(chalk.red("No tool configured"));
            return "Sorry no configured tool to answer the question";
          }
          const agruments = Object.values(input.arguments);

          log(chalk.blue(`Invoking the ${input.name} tool`));
          return choosenTool.invoke({
            a: agruments[0],
            b: agruments[1],
          });
        },
      })
    );
  const res = await chain.invoke({
    input: question,
    rendered_tools: renderedTools,
  });

  log(chalk.bgMagenta.white.bold(`The final result from the LLM is: `));
  log(chalk.blue(`Question:  ${question} `));
  log(chalk.blue(`The final result from the LLM is: ${res}`));
}

init("what is color of moon");
