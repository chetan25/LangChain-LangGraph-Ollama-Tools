import { ChatPromptTemplate } from "@langchain/core/prompts";

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

export const llmSystemPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are an helpful assitant that answers users question. Please keep the answers
      short and precise. No extra summary or explanation needed".
        {question}
      `,
  ],
]);

export const prompt = ChatPromptTemplate.fromMessages([
  ["system", systemPrompt],
  ["user", "{input}"],
]);
