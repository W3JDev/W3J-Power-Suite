
import { GoogleGenAI } from "@google/genai";

// This function creates a new instance for each call to ensure the latest API key is used,
// especially important for features like Veo that use a key selection dialog.
const getAi = () => new GoogleGenAI({ apiKey: process.env.API_KEY });

export default getAi;
