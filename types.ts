

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
  personaId?: string;
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

export interface SearchResult {
  type: 'Chat' | 'Media' | 'Scheduler';
  id: string;
  title: string;
  snippet: string;
  onClick: () => void;
}

export interface ScheduledItem {
  id: string;
  title: string;
  notes: string;
  dateTime: string; // ISO string format
  isComplete: boolean;
  createdAt: number;
}
