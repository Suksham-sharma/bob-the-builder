import express from "express";
import { ChatOpenAI } from "@langchain/openai";
import { ConversationSummaryMemory } from "langchain/memory";
import { LLMChain } from "langchain/chains";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
  PromptTemplate,
} from "@langchain/core/prompts";
import cors from "cors";
import dotenv from "dotenv";
import { BASE_PROMPT, getSystemPrompt } from "./prompts";
import { basePrompt as nodeBasePrompt } from "./defaults/node";
import { basePrompt as reactBasePrompt } from "./defaults/react";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 1,
  openAIApiKey: process.env.OPENAI_API_KEY,
});

const memory = new ConversationSummaryMemory({
  memoryKey: "chat_history",
  llm: model,
  inputKey: "input",
  outputKey: "text",
  returnMessages: true,
});

app.post("/template", async (req, res) => {
  const prompt = req.body.prompt;

  try {
    console.log(prompt);
    const response = await model.invoke([
      new SystemMessage(
        "Return either node or react based on what this project should be. Only return a single word either 'node' or 'react'. Do not return anything extra"
      ),
      new HumanMessage(prompt),
    ]);

    const answer = (response.content as string).toLowerCase().trim();
    console.log(answer);

    if (answer === "react") {
      res.json({
        prompts: [
          BASE_PROMPT,
          `Here is an artifact that contains all files of the project visible to you.\nConsider the contents of ALL files in the project.\n\n${reactBasePrompt}\n\nHere is a list of files that exist on the file system but are not being shown to you:\n\n  - .gitignore\n  - package-lock.json\n`,
        ],
        uiPrompts: [reactBasePrompt],
      });
      return;
    }

    if (answer === "node") {
      res.json({
        prompts: [
          `Here is an artifact that contains all files of the project visible to you.\nConsider the contents of ALL files in the project.\n\n${nodeBasePrompt}\n\nHere is a list of files that exist on the file system but are not being shown to you:\n\n  - .gitignore\n  - package-lock.json\n`,
        ],
        uiPrompts: [nodeBasePrompt],
      });
      return;
    }

    res.status(403).json({ message: "Cannot determine project type" });
  } catch (error: any) {
    res.status(500).json({
      message: "Error processing template request",
      error: error.message,
    });
  }
});

app.post("/chat", async (req, res) => {
  const message = req.body;
  console.log(message);

  try {
    const systemPrompt = getSystemPrompt();

    const prompt = ChatPromptTemplate.fromMessages([
      new SystemMessage(systemPrompt),
    ]);

    const flatInput =
      message.prompts.join("\n") +
      "\n" +
      message.uiPrompts.join("\n") +
      "\n" +
      message.userPrompt.join("\n");

    const chain = new LLMChain({
      llm: model,
      prompt: prompt,
    });

    try {
      console.log("Flat Input:", flatInput);
      const response = await chain.invoke({ input: flatInput });
      console.log("Raw Chain Response:", JSON.stringify(response, null, 2));
      res.json(response);
    } catch (error: any) {
      console.error("Invocation error:", error.message);
      res
        .status(500)
        .json({ message: "Error processing request", error: error.message });
    }
  } catch (error: any) {
    console.log(error.message);
    res
      .status(500)
      .json({ message: "Error processing chat request", error: error.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
