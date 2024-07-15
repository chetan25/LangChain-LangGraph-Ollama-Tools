import { ChatOllama } from "@langchain/community/chat_models/ollama";

// create the model instance
export const model = new ChatOllama({ model: "mistral" });
