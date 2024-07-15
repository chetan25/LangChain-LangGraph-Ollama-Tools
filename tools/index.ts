import {
  DynamicStructuredTool,
  StructuredToolInterface,
  tool,
} from "@langchain/core/tools";
import { renderTextDescription } from "langchain/tools/render";
import { z } from "zod";

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
export const addTool = tool(
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
export const multiplyTool = new DynamicStructuredTool({
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

export const tools = [addTool, multiplyTool];
export const renderedTools = renderTextDescription(tools);

export const toolMap: Record<string, StructuredToolInterface> =
  Object.fromEntries(tools.map((tool) => [tool.name, tool]));

export const tooList = Object.keys(toolMap);
