// FIX: Import React to resolve 'Cannot find namespace 'React'' error when using React.FC.
import React from 'react';

export interface ChatMessagePart {
  text?: string;
  inlineData?: {
    data: string; // base64 string
    mimeType: string;
  };
}

export interface ChatMessage {
  role: 'user' | 'model';
  parts: ChatMessagePart[];
  groundingChunks?: GroundingChunk[];
  suggestions?: string[];
}

// New type for a single chat session
export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
}

// New type for TTS configuration
export interface AudioConfig {
  voice: string | null;
  isPlaying: string | null; // Will hold the message index that is playing
}


export interface GroundingChunk {
  web?: {
    uri: string;
    title: string;
  };
  maps?: {
    uri: string;
    title: string;
    placeAnswerSources?: {
      reviewSnippets: {
        text: string;
        authorName: string;
      }[];
    };
  };
}

export interface Persona {
  id: string;
  name: string;
  description: string;
  prompt: string;
  voiceId: 'Kore' | 'Puck' | 'Zephyr' | 'Charon';
  isCustom?: boolean;
}
